<#
  SQLShare rerun: setup, checks, and the run, for Windows 11 PowerShell (5.1 or 7).

  Run from the folder that contains this file, in a normal (not Administrator) window:
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Check       # 1-minute computer check
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -SelfTest    # installation test, a few minutes
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick       # scaled-down run on the real data
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1              # the full run, both protocols
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Ablation    # after the full run: one change at a time

  Options: -Protocol paper|corrected|both   -Zip <path to sqlshare_data_release1.zip>
           -Workers N   -Threads N   -Serial (= -Workers 1 -Threads 1)   -MemoryLimit 8GB
           -PipIndexUrl https://pypi.tuna.tsinghua.edu.cn/simple   (a faster package mirror in China)
           -SkipTests   -AblationMode both|add|drop   -Replicates N (sampling-seed sets, default 2)

  Safe to stop (Ctrl+C) and start again: finished stages are skipped, the download resumes.
  Logs: logs\ (this script, tests, computer check) and results\run.log (the run itself).
#>
param(
    [switch]$SelfTest,
    [switch]$Quick,
    [switch]$Check,
    [ValidateSet("both", "paper", "corrected")][string]$Protocol = "both",
    [string]$Zip = "",
    [int]$Workers = 0,
    [int]$Threads = 0,
    [switch]$Serial,
    [string]$MemoryLimit = "",
    [string]$PipIndexUrl = "",
    [switch]$SkipTests,
    [switch]$Ablation,
    [ValidateSet("both", "add", "drop")][string]$AblationMode = "both",
    [ValidateRange(0, 10)][int]$Replicates = 2
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$onWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or ($IsWindows -eq $true)
$logs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = $false
try { Start-Transcript -Path (Join-Path $logs "run_all_$stamp.txt") -Append | Out-Null; $transcript = $true } catch { }

function Say([string]$text, [string]$color = "Cyan") { Write-Host $text -ForegroundColor $color }

$consoleRestore = $null
if ($onWindows) {
    # A click inside an old-style console window ("QuickEdit") pauses every program that
    # writes to it until a key is pressed. Switch that off while this script runs (it is
    # switched back on at the end, so text can be selected and copied again).
    try {
        Add-Type -Namespace SqlShareRerun -Name ConsoleMode -ErrorAction Stop -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
'@
        $hIn = [SqlShareRerun.ConsoleMode]::GetStdHandle(-10)
        $mode = [uint32]0
        if ([SqlShareRerun.ConsoleMode]::GetConsoleMode($hIn, [ref]$mode)) {
            $newMode = $mode
            if (($newMode -band 0x40) -ne 0) { $newMode = $newMode - 0x40 }
            $newMode = $newMode -bor 0x80
            if ([SqlShareRerun.ConsoleMode]::SetConsoleMode($hIn, [uint32]$newMode)) { $consoleRestore = @($hIn, $mode) }
        }
    } catch { }
    try {
        $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
        if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            Write-Host "Note: this window runs as administrator. Please use a normal PowerShell window for every step." -ForegroundColor Yellow
        }
    } catch { }
}

function Invoke-Probe([string]$Exe, [string]$Arguments, [int]$TimeoutSec = 60) {
    # Run a program with no input and a time limit; returns $null if it cannot start or hangs.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    try { $p = [System.Diagnostics.Process]::Start($psi) } catch { return $null }
    try { $p.StandardInput.Close() } catch { }
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch { }
        return $null
    }
    $p.WaitForExit()
    return [pscustomobject]@{ Code = $p.ExitCode; Out = $out.Result.Trim(); Err = $err.Result.Trim() }
}

$probeCode = "import sys, struct; print('%d.%d %d %s' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8, sys.executable))"

function Test-Python([string]$Exe, [string]$Prefix) {
    # Returns the interpreter path if $Exe is 64-bit Python 3.11, 3.12 or 3.13, else $null.
    $argText = ("$Prefix -c `"$probeCode`"").Trim()
    $r = Invoke-Probe $Exe $argText 60
    if ($null -eq $r -or $r.Code -ne 0) { return $null }
    $parts = $r.Out -split ' ', 3
    if ($parts.Count -eq 3 -and ($parts[0] -in @("3.11", "3.12", "3.13")) -and $parts[1] -eq "64") { return $parts[2] }
    return $null
}

function Find-Python {
    if ($onWindows) {
        foreach ($v in @("3.12", "3.13", "3.11")) {
            $exe = Test-Python "py" "-$v"
            if ($exe) { return $exe }
        }
        foreach ($name in @("python", "python3")) {
            $exe = Test-Python $name ""
            if ($exe) { return $exe }
        }
    } else {
        foreach ($name in @("python3.12", "python3.13", "python3.11", "python3")) {
            $exe = Test-Python $name ""
            if ($exe) { return $exe }
        }
    }
    return $null
}

$code = 1
try {
    if ($Serial) { $Workers = 1; $Threads = 1 }

    # ---- 1. Python and the virtual environment ------------------------------------------------
    Say "== 1/5 Python"
    if ($onWindows) { $vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" } else { $vpy = Join-Path $PSScriptRoot ".venv/bin/python" }
    $venvOk = $false
    if (Test-Path -LiteralPath $vpy) { $venvOk = [bool](Test-Python $vpy "") }
    if (-not $venvOk) {
        $pyExe = Find-Python
        if (-not $pyExe) {
            Say "64-bit Python 3.12 (or 3.11, 3.13) was not found. Python 3.14 does not work yet." "Red"
            Say "Install it with:  winget install -e --id Python.Python.3.12"
            Say "then close this window, open a new PowerShell window, and run this script again."
            exit 1
        }
        Say "using $pyExe"
        if (Test-Path -LiteralPath ".venv") { Remove-Item -LiteralPath ".venv" -Recurse -Force }
        & $pyExe -m venv .venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $vpy)) { Say "could not create .venv" "Red"; exit 1 }
    } else {
        Say "using the existing .venv"
    }

    # ---- 2. packages -------------------------------------------------------------------------------
    Say "== 2/5 Packages"
    & $vpy -m sqlshare_rerun.pins
    if ($LASTEXITCODE -ne 0) {
        Say "installing the pinned packages (about 100 MB; progress is shown below)"
        $pipArgs = @("-m", "pip", "install", "--only-binary=:all:", "--timeout", "60", "--retries", "10", "-r", "requirements.txt")
        if ($PipIndexUrl) { $pipArgs += @("--index-url", $PipIndexUrl) }
        & $vpy @pipArgs
        if ($LASTEXITCODE -ne 0) {
            Say "pip could not install the packages. If the download is slow or blocked, run again with:" "Red"
            Say "  -PipIndexUrl https://pypi.tuna.tsinghua.edu.cn/simple"
            exit 1
        }
        & $vpy -m sqlshare_rerun.pins
        if ($LASTEXITCODE -ne 0) { Say "the installed packages do not match requirements.txt" "Red"; exit 1 }
    } else {
        Say "all pinned packages are installed"
    }

    # ---- 3. computer check -----------------------------------------------------------------------
    Say "== 3/5 Computer check (about 1 minute)"
    $docArgs = @("-m", "sqlshare_rerun", "doctor")
    if ($Zip) { $docArgs += @("--zip", $Zip) }
    if (-not $SelfTest -and -not $Check) { $docArgs += "--for-full-run" }
    $docFile = Join-Path $logs "doctor.json"
    Remove-Item -LiteralPath $docFile -Force -ErrorAction SilentlyContinue
    & $vpy @docArgs
    $docCode = $LASTEXITCODE
    if (($docCode -ne 0 -and $docCode -ne 1) -or -not (Test-Path -LiteralPath $docFile)) {
        Say "The computer check stopped unexpectedly (code $docCode); see the output above and send the logs folder." "Red"
        exit 1
    }
    $doc = Get-Content -Raw -LiteralPath $docFile | ConvertFrom-Json
    if ($doc.fail) { Say "The computer check found a problem (FAIL lines above). Fix it and run again." "Red"; exit 1 }
    if ($Workers -le 0 -and $doc.recommend.workers) { $Workers = [int]$doc.recommend.workers; Say "using -Workers $Workers (from the computer check)" "Yellow" }
    if ($Threads -le 0 -and $doc.recommend.threads) { $Threads = [int]$doc.recommend.threads; Say "using -Threads $Threads (from the computer check)" "Yellow" }
    if ($Check) { $code = 0; exit 0 }

    # ---- 4. tests ----------------------------------------------------------------------------------
    if (-not $SkipTests) {
        Say "== 4/5 Tests: unit tests, checks against the legacy code, safety checks (2 to 6 minutes)"
        if ($Workers -eq 1) { $env:SQLSHARE_TEST_WORKERS = "1" } else { $env:SQLSHARE_TEST_WORKERS = "2" }
        if ($Threads -eq 1) { $env:SQLSHARE_TEST_THREADS = "1" } else { $env:SQLSHARE_TEST_THREADS = "4" }
        $ptArgs = @("-m", "pytest", "-q", "--junitxml=logs/tests.xml")
        if (Test-Path -LiteralPath ".pytest_tmp") {
            Remove-Item -LiteralPath ".pytest_tmp" -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath ".pytest_tmp") {
                $alt = ".pytest_tmp_$stamp"
                Say "The old test folder .pytest_tmp cannot be removed (probably made from an administrator window); using $alt." "Yellow"
                $ptArgs += "--basetemp=$alt"
            }
        }
        & $vpy @ptArgs
        if ($LASTEXITCODE -ne 0) {
            $hang = Join-Path $logs "test_hang.txt"
            if ((Test-Path -LiteralPath $hang) -and ((Get-Item -LiteralPath $hang).Length -gt 0)) {
                Say "A test ran too long and was stopped:" "Red"
                Get-Content -LiteralPath (Join-Path $logs "current_test.txt") -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
                Get-Content -LiteralPath $hang -Tail 40 | ForEach-Object { Write-Host $_ }
            }
            Say "Tests failed. Please send the whole logs folder." "Red"
            exit 1
        }
    } else {
        Say "== 4/5 Tests skipped (-SkipTests)"
    }

    # ---- 5. the run --------------------------------------------------------------------------------
    $common = @()
    if ($Workers -gt 0) { $common += @("--workers", "$Workers") }
    if ($Threads -gt 0) { $common += @("--threads", "$Threads") }
    if ($MemoryLimit) { $common += @("--memory-limit", $MemoryLimit) }
    if ($SelfTest) {
        Say "== 5/5 Self-test on a small synthetic release (1 to 5 minutes)"
        & $vpy -m sqlshare_rerun selftest @common
        $code = $LASTEXITCODE
        if ($code -eq 0) { Say "SELF-TEST PASSED. You can start the real run now." "Green" } else { Say "Self-test failed. Please send selftest_run\results\run.log and the logs folder." "Red" }
        exit $code
    }
    if ($Ablation) {
        Say "== 5/5 Ablation on the main market (mode: $AblationMode, seed sets: $Replicates). Several hours; keep the computer plugged in and awake."
        Say "Progress: results\run.log. Ctrl+C is safe; the same command continues where it stopped."
        $ablArgs = @("-m", "sqlshare_rerun", "ablation", "--mode", $AblationMode, "--replicates", "$Replicates") + $common
        if ($Quick) { $ablArgs += "--quick" }
        if ($Zip) { $ablArgs += @("--zip", $Zip) }
        & $vpy @ablArgs
        $code = $LASTEXITCODE
        if ($code -eq 0) { Say "FINISHED. The ablation report is results\ablation\ABLATION.md" "Green" } else { Say "The ablation stopped (code $code). Running the same command again continues where it stopped. Please send results\run.log." "Red" }
        exit $code
    }
    $driveName = (Get-Item -LiteralPath $PSScriptRoot).PSDrive.Name
    $freeGB = [math]::Round((Get-PSDrive -Name $driveName).Free / 1GB, 1)
    if ($freeGB -lt 15) { Say "Only $freeGB GB free on drive $driveName. The full run needs about 15 GB." "Red"; exit 1 }
    Say "== 5/5 SQLShare rerun (protocol: $Protocol). Keep the computer plugged in and awake."
    Say "Progress: results\run.log. If it ever looks stuck, send results\run.log and results\stack_dumps.log."
    $runArgs = @("-m", "sqlshare_rerun", "all", "--protocol", $Protocol) + $common
    if ($Quick) { $runArgs += "--quick" }
    if ($Zip) { $runArgs += @("--zip", $Zip) }
    & $vpy @runArgs
    $code = $LASTEXITCODE
    if ($code -eq 0) { Say "FINISHED. The main report is results\REPORT.md" "Green" } else { Say "The run stopped (code $code). Running the same command again continues where it stopped. Please send results\run.log." "Red" }
    exit $code
}
finally {
    if ($consoleRestore) { try { [void][SqlShareRerun.ConsoleMode]::SetConsoleMode($consoleRestore[0], [uint32]$consoleRestore[1]) } catch { } }
    if ($transcript) { try { Stop-Transcript | Out-Null } catch { } }
}
