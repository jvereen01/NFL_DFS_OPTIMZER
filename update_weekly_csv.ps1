# Replaces last week's FanDuel players-list CSV with this week's and pushes to main.
# Usage: .\update_weekly_csv.ps1 "C:\Users\jamin\Downloads\FanDuel-NFL-2026 EDT-09 EDT-20 EDT-141207-players-list.csv"
param(
    [Parameter(Mandatory = $true)]
    [string]$NewCsvPath
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $NewCsvPath)) {
    throw "File not found: $NewCsvPath"
}

# Remove any old players-list CSVs tracked in the repo
Get-ChildItem -Path $PSScriptRoot -Filter "*players-list*.csv" | ForEach-Object {
    git rm --quiet $_.Name
}

# Copy in the new file and stage it
$destName = Split-Path $NewCsvPath -Leaf
Copy-Item $NewCsvPath -Destination (Join-Path $PSScriptRoot $destName)
git add $destName

git commit -m "Update weekly player CSV: $destName"
git push origin main

Write-Host "Deployed. Streamlit Cloud will auto-redeploy with the new CSV."
