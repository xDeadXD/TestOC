# Builds the .NET sensor sidecar (self-contained single file) and copies it to
# src-tauri/binaries/testoc-sensors-<host triple>.exe for Tauri's externalBin.
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"
$root = Resolve-Path "$PSScriptRoot\.."
$proj = Join-Path $root 'sensors-sidecar\TestOC.Sensors.csproj'
$out = Join-Path $root 'sensors-sidecar\publish'
dotnet publish $proj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -o $out --nologo
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }
$triple = ((rustc -vV) | Select-String '^host: (.+)$').Matches[0].Groups[1].Value
$dest = Join-Path $root "src-tauri\binaries\testoc-sensors-$triple.exe"
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
Copy-Item (Join-Path $out 'testoc-sensors.exe') $dest -Force
Write-Host "sidecar -> $dest ($([math]::Round((Get-Item $dest).Length / 1MB, 1)) MB)"
