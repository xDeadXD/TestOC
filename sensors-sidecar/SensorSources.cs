using LibreHardwareMonitor.Hardware;
using Microsoft.Win32;

namespace TestOC.Sensors;

public interface ISensorSource : IDisposable
{
    /// <summary>Read all published sensors. Sets <see cref="Status"/> / <see cref="StatusText"/>.</summary>
    IReadOnlyList<SensorRec> Read();
    uint Status { get; }
    string StatusText { get; }
}

public static class Kinds
{
    public const byte HwOther = 0, HwCpu = 1, HwMotherboard = 2, HwSuperIo = 3, HwGpu = 4;
    public const byte Other = 0, Temperature = 1, Clock = 2, Power = 3, Voltage = 4, Fan = 5, Load = 6, Factor = 7, SmallData = 8;

    public static byte Hw(HardwareType t) => t switch
    {
        HardwareType.Cpu => HwCpu,
        HardwareType.Motherboard => HwMotherboard,
        HardwareType.SuperIO => HwSuperIo,
        HardwareType.GpuNvidia or HardwareType.GpuAmd or HardwareType.GpuIntel => HwGpu,
        _ => HwOther,
    };

    public static byte Sensor(SensorType t) => t switch
    {
        SensorType.Temperature => Temperature,
        SensorType.Clock => Clock,
        SensorType.Power => Power,
        SensorType.Voltage => Voltage,
        SensorType.Fan => Fan,
        SensorType.Load => Load,
        SensorType.Factor => Factor,
        SensorType.SmallData => SmallData,
        _ => Other,
    };
}

/// <summary>
/// Read-only LibreHardwareMonitor source: CPU + motherboard/SuperIO + universal GPU fallback
/// (enables AMD Radeon & Intel Arc GPU sensors when NVML is not present). Never calls control APIs.
/// </summary>
public sealed class LhmSource : ISensorSource
{
    private readonly Computer _computer;
    private readonly List<SensorRec> _buf = new(256);
    // Cache Identifier.ToString() per ISensor so steady-state polling allocates zero managed strings.
    private readonly Dictionary<ISensor, string> _identCache = new(256);

    public uint Status { get; private set; } = Layout.StatusInitializing;
    public string StatusText { get; private set; } = "";

    public LhmSource(bool enableGpu = true)
    {
        _computer = new Computer
        {
            IsCpuEnabled = true,
            IsMotherboardEnabled = true,
            IsGpuEnabled = enableGpu,
            IsMemoryEnabled = false,
            IsStorageEnabled = false,
            IsNetworkEnabled = false,
            IsControllerEnabled = false,
            IsPsuEnabled = false,
            IsBatteryEnabled = false,
        };
        _computer.Open();
    }

    public static bool PawnIoInstalled()
    {
        using var key = Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Services\PawnIO");
        return key != null;
    }

    public IReadOnlyList<SensorRec> Read()
    {
        _buf.Clear();
        foreach (IHardware hw in _computer.Hardware)
        {
            hw.Update();
            Visit(hw, Kinds.Hw(hw.HardwareType));
        }
        bool cpuTemp = false;
        for (int i = 0; i < _buf.Count; i++)
        {
            var r = _buf[i];
            if (r.HwKind == Kinds.HwCpu && r.SensorKind == Kinds.Temperature && float.IsFinite(r.Value))
            {
                cpuTemp = true;
                break;
            }
        }
        if (cpuTemp)
        {
            Status = Layout.StatusOk;
            StatusText = "";
        }
        else if (!PawnIoInstalled())
        {
            Status = Layout.StatusPawnIoMissing;
            StatusText = "PawnIO driver not installed: CPU temperatures unavailable";
        }
        else
        {
            Status = Layout.StatusOk;
            StatusText = "No CPU temperature sensor exposed by LibreHardwareMonitor";
        }
        return _buf;
    }

    private void Visit(IHardware hw, byte hwKind)
    {
        foreach (IHardware sub in hw.SubHardware)
        {
            sub.Update();
            Visit(sub, sub.HardwareType == HardwareType.SuperIO ? Kinds.HwSuperIo : hwKind);
        }
        foreach (ISensor s in hw.Sensors)
        {
            byte kind = Kinds.Sensor(s.SensorType);
            if (kind == Kinds.Other || hwKind == Kinds.HwOther || s.Value is not float v) continue;
            if (_buf.Count >= Layout.Capacity) return;
            if (!_identCache.TryGetValue(s, out string? ident))
            {
                ident = s.Identifier.ToString();
                _identCache[s] = ident;
            }
            _buf.Add(new SensorRec(hwKind, kind, (ushort)s.Index, v, s.Min ?? v, s.Max ?? v, ident, s.Name));
        }
    }

    public void Dispose() => _computer.Close();
}

/// <summary>
/// Synthetic Zen 3-like sensors for automated tests (no LHM, no elevation).
/// <c>--fake-ramp</c> raises Tctl linearly from 50 °C by 1 °C/s to exercise the watchdog.
/// </summary>
public sealed class FakeSource : ISensorSource
{
    private readonly DateTime _start = DateTime.UtcNow;
    private readonly bool _ramp;
    private readonly float _base;

    public FakeSource(bool ramp, float baseTemp)
    {
        _ramp = ramp;
        _base = baseTemp;
    }

    public uint Status => Layout.StatusOk;
    public string StatusText => "fake sensors";

    public IReadOnlyList<SensorRec> Read()
    {
        double t = (DateTime.UtcNow - _start).TotalSeconds;
        float tctl = _ramp ? (float)(_base + t) : (float)(_base + 6 * Math.Sin(t / 8));
        var list = new List<SensorRec>
        {
            new(Kinds.HwCpu, Kinds.Temperature, 0, tctl, tctl, tctl, "/amdcpu/0/temperature/2", "Core (Tctl/Tdie)"),
            new(Kinds.HwCpu, Kinds.Temperature, 1, tctl - 2, tctl - 2, tctl - 2, "/amdcpu/0/temperature/3", "CCD1 (Tdie)"),
            new(Kinds.HwCpu, Kinds.Power, 0, (float)(60 + 20 * Math.Sin(t / 5)), 0, 120, "/amdcpu/0/power/0", "Package"),
            new(Kinds.HwCpu, Kinds.Voltage, 0, 1.25f, 1.2f, 1.4f, "/amdcpu/0/voltage/0", "Core (SVI2 TFN)"),
            new(Kinds.HwSuperIo, Kinds.Temperature, 0, 36f, 30, 40, "/lpc/nct6798d/temperature/0", "System"),
            new(Kinds.HwSuperIo, Kinds.Fan, 0, (float)(1100 + 100 * Math.Sin(t)), 0, 2000, "/lpc/nct6798d/fan/0", "CPU Fan"),
        };
        for (int c = 1; c <= 8; c++)
        {
            float mhz = (float)(4400 + 150 * Math.Sin(t + c));
            list.Add(new(Kinds.HwCpu, Kinds.Clock, (ushort)c, mhz, 550, 4650, $"/amdcpu/0/clock/{c}", $"Core #{c}"));
            list.Add(new(Kinds.HwCpu, Kinds.Voltage, (ushort)c, 1.3f, 1.0f, 1.45f, $"/amdcpu/0/voltage/{c}", $"Core #{c} VID"));
        }
        return list;
    }

    public void Dispose() { }
}
