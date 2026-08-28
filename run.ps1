<#
.SYNOPSIS
    Windows equivalent of the Makefile. Same target names.

.EXAMPLE
    .\run.ps1 up
    .\run.ps1 ingest
    .\run.ps1 eval
    .\run.ps1 ask "What are the red flags for trade-based money laundering?"
#>
param(
    [Parameter(Position = 0)][string]$Target = 'help',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-App {
    param([string[]]$ModuleArgs)
    docker compose run --rm app python -m @ModuleArgs
    if ($LASTEXITCODE -ne 0) { throw "failed: python -m $($ModuleArgs -join ' ')" }
}

function Get-PgUser { if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { 'aml' } }
function Get-PgDb   { if ($env:POSTGRES_DB)   { $env:POSTGRES_DB }   else { 'aml' } }

switch ($Target) {

    'help' {
        Write-Host ''
        Write-Host '  up             Start Postgres + app, wait until healthy'
        Write-Host '  down           Stop containers, keep the data volume'
        Write-Host '  nuke           Stop containers AND destroy the data volume'
        Write-Host '  migrate        Reapply migrations against a running database'
        Write-Host '  psql           Open a psql shell'
        Write-Host '  logs           Tail application logs'
        Write-Host '  fetch          Download every document in corpus/manifest.yaml'
        Write-Host '  ingest         Download, extract, chunk, embed, load'
        Write-Host '  corpus-stats   Document and chunk counts per profile'
        Write-Host '  eval           Regenerate every number in the README'
        Write-Host '  eval-retrieval M1 retrieval metrics'
        Write-Host '  eval-answers   M4 answer-quality metrics'
        Write-Host '  report         Rewrite README tables from results/'
        Write-Host '  serve          Run the API in the foreground'
        Write-Host '  ask "<q>"      Ask one question end to end'
        Write-Host '  test           Run the test suite'
        Write-Host ''
    }

    'up' {
        docker compose up -d --build
        if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }
        Write-Host 'waiting for postgres...'
        $u = Get-PgUser; $d = Get-PgDb
        for ($i = 0; $i -lt 60; $i++) {
            docker compose exec -T db pg_isready -U $u -d $d 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host 'up. postgres on localhost:5434, api on localhost:8000'
                exit 0
            }
            Start-Sleep -Seconds 1
        }
        throw 'postgres did not become healthy within 60s'
    }

    'down'  { docker compose down }
    'nuke'  { docker compose down -v }
    'logs'  { docker compose logs -f app }
    'psql'  { docker compose exec db psql -U (Get-PgUser) -d (Get-PgDb) }
    'serve' { docker compose up app }

    'migrate' {
        $u = Get-PgUser; $d = Get-PgDb
        foreach ($f in Get-ChildItem migrations\*.sql | Sort-Object Name) {
            Write-Host "applying $($f.Name)"
            Get-Content $f.FullName -Raw | docker compose exec -T db psql -v ON_ERROR_STOP=1 -U $u -d $d
            if ($LASTEXITCODE -ne 0) { throw "migration failed: $($f.Name)" }
        }
    }

    'fetch'          { Invoke-App @('aml_agent.ingest.download') }
    'ingest'         { Invoke-App @('aml_agent.ingest.pipeline') }
    'corpus-stats'   { Invoke-App @('aml_agent.ingest.stats') }
    'eval-retrieval' { Invoke-App @('aml_agent.evaluation.retrieval') }
    'eval-answers'   { Invoke-App @('aml_agent.evaluation.answers') }
    'report'         { Invoke-App @('aml_agent.evaluation.report') }

    'eval' {
        Invoke-App @('aml_agent.evaluation.retrieval')
        Invoke-App @('aml_agent.evaluation.answers')
    }

    'ask' {
        if (-not $Rest -or $Rest.Count -eq 0) { throw 'usage: .\run.ps1 ask "your question"' }
        Invoke-App (@('aml_agent.agent.cli') + $Rest)
    }

    'test' {
        docker compose run --rm app python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
    }

    default { throw "unknown target '$Target'. Run .\run.ps1 help" }
}
