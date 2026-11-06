# Minimal stage telemetry: wall, process CPU, VmRSS/VmHWM.
function memory_kib()
    isfile("/proc/self/status") || return (-1, Int(Sys.maxrss() ÷ 1024))   # non-Linux: peak RSS only
    s = read("/proc/self/status", String)
    g(k) = (m = match(Regex("(?m)^" * k * ":\\s*(\\d+)"), s); m === nothing ? -1 : parse(Int, m[1]))
    (g("VmRSS"), g("VmHWM"))
end

struct _TS; s::Int64; ns::Int64; end
function cpu_seconds()
    if Sys.iswindows()
        c = Ref{UInt64}(0); e = Ref{UInt64}(0); k = Ref{UInt64}(0); u = Ref{UInt64}(0)
        h = ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ())
        ok = ccall((:GetProcessTimes, "kernel32"), stdcall, Cint,
                   (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}), h, c, e, k, u)
        return ok == 0 ? NaN : (k[] + u[]) * 1e-7
    end
    Sys.islinux() || return NaN
    x = Ref(_TS(0, 0))
    ccall(:clock_gettime, Cint, (Cint, Ref{_TS}), 2, x) == 0 || return NaN
    x[].s + x[].ns * 1e-9
end

const STAGE_LOCK = ReentrantLock()      # concurrent blocks share one stages.tsv

mutable struct Stage
    path::String
end

"""Run f(), append a TSV row (label, wall, cpu, busy cores, rss/hwm after) to st.path."""
function stage!(f, st::Stage, label::AbstractString; detail="")
    c0 = cpu_seconds(); t0 = time(); _, h0 = memory_kib()
    v = f()
    wall = time() - t0; cpu = cpu_seconds() - c0; rss, hwm = memory_kib()
    lock(STAGE_LOCK); try
    fresh = !isfile(st.path)
    open(st.path, "a") do io
        fresh && println(io, "label\twall_s\tcpu_s\tbusy_cores\trss_kib\thwm_kib\thwm_before_kib\tthreads\tblas\tdetail")
        println(io, join((label, round(wall; digits=4), round(cpu; digits=3), round(cpu / max(wall, 1e-9); digits=2),
                          rss, hwm, h0, Threads.nthreads(), BLAS.get_num_threads(), replace(string(detail), r"[\t\n]" => " ")), "\t"))
    end
    println("STAGE ", label, " wall=", round(wall; digits=3), " cpu=", round(cpu; digits=1), " hwm_kib=", hwm, " ", detail); flush(stdout)
    finally; unlock(STAGE_LOCK); end
    v
end

"""
    interleave_memory!(nodes=2)

Set MPOL_INTERLEAVE over `nodes` NUMA nodes for every Julia thread (Linux set_mempolicy).
New pages are then spread over both sockets' memory controllers instead of the node of the
first-touching thread.  Returns the per-thread syscall results (0 = success).
"""
function interleave_memory!(nodes::Int=2)
    Sys.islinux() || return Int[]
    SYS_set_mempolicy = 238          # x86_64
    MPOL_INTERLEAVE = 3
    mask = Ref{Culong}((Culong(1) << nodes) - 1)
    res = zeros(Int, Threads.nthreads())
    set1() = ccall(:syscall, Clong, (Clong, Cint, Ptr{Culong}, Culong), SYS_set_mempolicy, MPOL_INTERLEAVE, mask, Culong(8 * sizeof(Culong)))
    Threads.@threads :static for i in 1:Threads.nthreads()
        res[Threads.threadid()] = set1()
    end
    set1()
    res
end
