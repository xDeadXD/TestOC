using System.Diagnostics;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime;
using System.Runtime.InteropServices;

namespace TestOC.Sensors;

/// <summary>
/// testoc-sensors.exe --map &lt;name&gt; --parent-pid &lt;pid&gt; [--no-gpu] [--fake [--fake-ramp] [--fake-base &lt;°C&gt;]] [--interval-ms 500]
///
/// Opens the mapping created by the (unelevated) TestOC app and publishes sensor frames at 2 Hz.
/// Exits when the parent exits, when the reader sets control=1, or if the mapping can't be opened.
/// </summary>
public static class Program
{
    private const uint QuotaLimitsHardWsMinEnable = 0x00000001;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetProcessWorkingSetSizeEx(IntPtr hProcess, nuint dwMinimumWorkingSetSize, nuint dwMaximumWorkingSetSize, uint flags);

    private static StreamWriter? _log;

    private static void Log(string msg)
    {
        try
        {
            _log?.WriteLine($"{DateTime.Now:HH:mm:ss.fff} {msg}");
            _log?.Flush();
        }
        catch { /* logging must never kill the sidecar */ }
    }

    private static void ProtectFromMemoryPressure()
    {
        // Run above stress workers and OS housekeeping so LHM core-affinity MSR hops never starve.
        try { Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High; } catch { }
        try { Thread.CurrentThread.Priority = ThreadPriority.Highest; } catch { }
        GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
    }

    private static void PinWorkingSet()
    {
        // Compact once after LHM initializes, then hard-lock the minimum working set (when elevated)
        // so 100% Absolut RAM stress cannot trim or page out our CLR/LHM pages to disk.
        try
        {
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            using var proc = Process.GetCurrentProcess();
            proc.Refresh();
            long currentWs = proc.WorkingSet64;
            nuint minWs = (nuint)Math.Clamp(currentWs + (24L << 20), 48L << 20, 192L << 20);
            nuint maxWs = minWs + (128u << 20);
            bool hard = SetProcessWorkingSetSizeEx(proc.Handle, minWs, maxWs, QuotaLimitsHardWsMinEnable);
            bool soft = hard || SetProcessWorkingSetSizeEx(proc.Handle, minWs, maxWs, 0);
            Log($"ws_pin hard={hard} soft={soft} min={minWs >> 20}MiB cur={currentWs >> 20}MiB prio={proc.PriorityClass}");
        }
        catch (Exception e)
        {
            Log($"ws_pin failed: {e.Message}");
        }
    }

    public static unsafe int Main(string[] args)
    {
        string? map = null;
        int parentPid = 0, intervalMs = 500;
        bool fake = false, ramp = false, enableGpu = true;
        float fakeBase = 50f;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--map": map = args[++i]; break;
                case "--parent-pid": parentPid = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--interval-ms": intervalMs = Math.Clamp(int.Parse(args[++i], CultureInfo.InvariantCulture), 100, 5000); break;
                case "--no-gpu": enableGpu = false; break;
                case "--fake": fake = true; break;
                case "--fake-ramp": ramp = true; break;
                case "--fake-base": fakeBase = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
            }
        }
        try
        {
            _log = new StreamWriter(Path.Combine(Path.GetTempPath(), "testoc-sensors.log"), append: false) { AutoFlush = true };
        }
        catch { }
        ProtectFromMemoryPressure();
        Log($"start pid={Environment.ProcessId} map={map} parent={parentPid} fake={fake} gpu={enableGpu} elevated={Environment.IsPrivilegedProcess}");
        if (map is null || parentPid <= 0)
        {
            Log("usage error");
            return 2;
        }

        Process parent;
        try { parent = Process.GetProcessById(parentPid); }
        catch (Exception e) { Log($"parent not found: {e.Message}"); return 3; }

        using (parent)
        {
            MemoryMappedFile mmf;
            try { mmf = MemoryMappedFile.OpenExisting(map, MemoryMappedFileRights.ReadWrite); }
            catch (Exception e) { Log($"open mapping failed: {e.Message}"); return 4; }

            using (mmf)
            using (var view = mmf.CreateViewAccessor(0, Layout.MappingSize, MemoryMappedFileAccess.ReadWrite))
            {
                byte* ptr = null;
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                try
                {
                    var w = new SharedMemoryWriter(ptr);
                    w.WriteHeaderStatic((uint)Environment.ProcessId);
                    w.WriteFrame(Layout.StatusInitializing, "starting LibreHardwareMonitor…", NowMs(), Array.Empty<SensorRec>());

                    ISensorSource src;
                    try
                    {
                        src = fake ? new FakeSource(ramp, fakeBase) : new LhmSource(enableGpu);
                    }
                    catch (Exception e)
                    {
                        Log($"sensor init failed: {e}");
                        w.WriteFrame(Layout.StatusError, $"sensor init failed: {e.Message}", NowMs(), Array.Empty<SensorRec>());
                        return 5;
                    }

                    using (src)
                    {
                        // Prime first sensor frame before locking working set so all LHM code/data pages are resident.
                        try
                        {
                            var warm = src.Read();
                            w.WriteFrame(src.Status, src.StatusText, NowMs(), warm);
                        }
                        catch (Exception e)
                        {
                            Log($"warmup read failed: {e.Message}");
                        }
                        PinWorkingSet();

                        var sw = Stopwatch.StartNew();
                        var readSw = new Stopwatch();
                        long next = intervalMs;
                        int errors = 0, slowLogs = 0;
                        while (true)
                        {
                            if (w.Control == 1) { Log("shutdown requested"); break; }
                            if (parent.WaitForExit(0)) { Log("parent exited"); break; }
                            try
                            {
                                readSw.Restart();
                                var recs = src.Read();
                                long readMs = readSw.ElapsedMilliseconds;
                                w.WriteFrame(src.Status, src.StatusText, NowMs(), recs);
                                if (readMs > 1000 && slowLogs++ < 10)
                                    Log($"slow read: {readMs} ms");
                                errors = 0;
                            }
                            catch (Exception e)
                            {
                                if (errors++ < 5) Log($"read failed: {e.Message}");
                                w.WriteFrame(Layout.StatusError, $"sensor read failed: {e.Message}", NowMs(), Array.Empty<SensorRec>());
                            }
                            long sleep = next - sw.ElapsedMilliseconds;
                            if (sleep > 0) Thread.Sleep((int)sleep);
                            next = sw.ElapsedMilliseconds + intervalMs;
                        }
                    }
                }
                finally
                {
                    view.SafeMemoryMappedViewHandle.ReleasePointer();
                }
            }
        }
        Log("exit");
        return 0;
    }

    private static ulong NowMs() => (ulong)DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
}
