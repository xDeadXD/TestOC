using System.Text.Unicode;

namespace TestOC.Sensors;

/// <summary>One sensor reading in the shared-memory record format.</summary>
public readonly record struct SensorRec(byte HwKind, byte SensorKind, ushort Index, float Value, float Min, float Max, string Identifier, string Name);

public static class Layout
{
    public const uint Magic = 0x4D534C53; // 'SLSM'
    public const ushort Version = 1;
    public const int HeaderSize = 256;
    public const int RecordSize = 128;
    public const int Capacity = 400;
    public const int MappingSize = HeaderSize + RecordSize * Capacity; // 51_456

    // Header offsets (keep in sync with src-tauri/src/sidecar/layout.rs)
    public const int OffMagic = 0, OffVersion = 4, OffHeaderSize = 6, OffRecordSize = 8, OffCapacity = 10, OffRecordCount = 12;
    public const int OffSeq = 16, OffHeartbeat = 24, OffWriterPid = 32, OffStatus = 36, OffControl = 40, OffStatusText = 48, StatusTextLen = 208;
    // Record offsets
    public const int RHwKind = 0, RSensorKind = 1, RIndex = 2, RValue = 4, RMin = 8, RMax = 12, RIdent = 16, RName = 72, RStrLen = 56;

    public const uint StatusInitializing = 0, StatusOk = 1, StatusPawnIoMissing = 2, StatusError = 3;
}

/// <summary>
/// Seqlock writer over the mapping created by the Rust app. Interlocked.Increment is a full fence,
/// so the reader (odd = in progress) never observes a torn frame.
/// </summary>
public sealed unsafe class SharedMemoryWriter
{
    private readonly byte* _p;

    public SharedMemoryWriter(byte* basePtr) => _p = basePtr;

    public void WriteHeaderStatic(uint pid)
    {
        *(ushort*)(_p + Layout.OffVersion) = Layout.Version;
        *(ushort*)(_p + Layout.OffHeaderSize) = Layout.HeaderSize;
        *(ushort*)(_p + Layout.OffRecordSize) = Layout.RecordSize;
        *(ushort*)(_p + Layout.OffCapacity) = Layout.Capacity;
        *(uint*)(_p + Layout.OffWriterPid) = pid;
        Volatile.Write(ref *(uint*)(_p + Layout.OffMagic), Layout.Magic); // publish last
    }

    /// <summary>Reader → writer control word (1 = shutdown requested).</summary>
    public uint Control => Volatile.Read(ref *(uint*)(_p + Layout.OffControl));

    public void WriteFrame(uint status, string text, ulong heartbeatMs, IReadOnlyList<SensorRec> recs)
    {
        ref long seq = ref *(long*)(_p + Layout.OffSeq);
        Interlocked.Increment(ref seq); // odd: write in progress
        int n = Math.Min(recs.Count, Layout.Capacity);
        for (int i = 0; i < n; i++)
        {
            byte* r = _p + Layout.HeaderSize + i * Layout.RecordSize;
            new Span<byte>(r, Layout.RecordSize).Clear();
            var s = recs[i];
            r[Layout.RHwKind] = s.HwKind;
            r[Layout.RSensorKind] = s.SensorKind;
            *(ushort*)(r + Layout.RIndex) = s.Index;
            *(float*)(r + Layout.RValue) = s.Value;
            *(float*)(r + Layout.RMin) = s.Min;
            *(float*)(r + Layout.RMax) = s.Max;
            PutStr(r + Layout.RIdent, Layout.RStrLen, s.Identifier);
            PutStr(r + Layout.RName, Layout.RStrLen, s.Name);
        }
        *(uint*)(_p + Layout.OffRecordCount) = (uint)n;
        *(ulong*)(_p + Layout.OffHeartbeat) = heartbeatMs;
        *(uint*)(_p + Layout.OffStatus) = status;
        PutStr(_p + Layout.OffStatusText, Layout.StatusTextLen, text);
        Interlocked.Increment(ref seq); // even: frame complete
    }

    /// <summary>Zero-allocation UTF-8, NUL-padded, truncated on a code-point boundary.</summary>
    private static void PutStr(byte* dst, int len, string s)
    {
        var span = new Span<byte>(dst, len);
        span.Clear();
        if (string.IsNullOrEmpty(s) || len <= 1) return;
        Utf8.FromUtf16(s.AsSpan(), span[..(len - 1)], out _, out _, replaceInvalidSequences: true, isFinalBlock: true);
    }
}
