param(
    [string]$Python = '',
    [switch]$Smoke,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'robotic-arm-pick-and-place'
if (-not $Python) {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv-bc\Scripts\python.exe'),
        (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-bc\Scripts\python.exe')
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Create .venv-bc first, or supply -Python with the full interpreter path.'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$scripts = @('collect_reach_grasp_dataset.py', 'train_reach_grasp_bc.py', 'evaluate_reach_grasp_bc.py',
             'collect_reach_grasp_dagger.py', 'train_reach_grasp_dagger.py', 'evaluate_reach_grasp_dagger.py')
foreach ($name in $scripts) {
    if (-not (Test-Path -LiteralPath (Join-Path $project $name) -PathType Leaf)) { throw "Missing entry point: $name" }
}
if ($VerifyOnly) {
    & $Python -X utf8 -B -c 'import mujoco, numpy, torch; print("runtime:", mujoco.__version__, numpy.__version__, torch.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed.' }
    Write-Host "Verified interpreter and six entry points: $Python"
    return
}

function Invoke-LearningStep {
    param([string]$Name, [string[]]$StepArgs, [string]$ResultPrefix)
    $stageLines = [System.Collections.Generic.List[string]]::new()
    & $Python -X utf8 -B (Join-Path $project $Name) @StepArgs | ForEach-Object {
        $stageLine = $_.ToString()
        $stageLines.Add($stageLine)
        Write-Host $stageLine
    }
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Name. Existing outputs are retained." }
    $resultLine = $stageLines | Where-Object { $_.StartsWith($ResultPrefix) } | Select-Object -Last 1
    if (-not $resultLine) { throw "Stage did not return $ResultPrefix : $Name" }
    return $resultLine.Substring($ResultPrefix.Length).Trim()
}

function Get-CompletedSplit {
    param([string]$Manifest)
    $splitPath = Join-Path (Split-Path -Parent $Manifest) 'split.json'
    if (-not (Test-Path -LiteralPath $splitPath -PathType Leaf)) {
        throw "No usable split produced; inspect retained manifest: $Manifest"
    }
    return $splitPath
}

$episodes, $validation, $epochs, $bcEpochs = 40, 8, 100, 100
$bcSeed, $labelSeed, $diagnosticSeed, $pairedSeed = 300, 400, 2000, 3000
$testEpisodes, $label = 20, 'workflow_reproduction'
if ($Smoke) {
    $episodes, $validation, $epochs, $bcEpochs, $testEpisodes = 8, 2, 2, 20, 2
    $bcSeed, $labelSeed, $diagnosticSeed, $pairedSeed = 9000, 9100, 9200, 9300
    $label = 'workflow_smoke'
}
$bcManifest = Invoke-LearningStep -Name $scripts[0] -StepArgs @('--episodes', "$episodes", '--seed', "$bcSeed", '--validation-episodes', "$validation", '--label', $label) -ResultPrefix 'manifest = '
$bcSplit = Get-CompletedSplit $bcManifest
$bcRun = Invoke-LearningStep -Name $scripts[1] -StepArgs @('--split', $bcSplit, '--epochs', "$bcEpochs", '--label', $label) -ResultPrefix 'saved = '
$bcCheckpoint = Join-Path $bcRun 'best.pt'
$bcEvaluation = Invoke-LearningStep -Name $scripts[2] -StepArgs @('--checkpoint', $bcCheckpoint, '--episodes', "$testEpisodes", '--seed', "$diagnosticSeed", '--label', $label) -ResultPrefix 'saved = '
$labelManifest = Invoke-LearningStep -Name $scripts[3] -StepArgs @('--checkpoint', $bcCheckpoint, '--episodes', "$episodes", '--seed', "$labelSeed", '--validation-episodes', "$validation", '--label', $label) -ResultPrefix 'manifest = '
$labelSplit = Get-CompletedSplit $labelManifest
$aggregatedRun = Invoke-LearningStep -Name $scripts[4] -StepArgs @('--dagger-split', $labelSplit, '--epochs', "$epochs", '--label', $label) -ResultPrefix 'saved = '
$pairedEvaluation = Invoke-LearningStep -Name $scripts[5] -StepArgs @('--checkpoint', (Join-Path $aggregatedRun 'best.pt'), '--episodes', "$testEpisodes", '--seed', "$pairedSeed", '--label', $label) -ResultPrefix 'saved = '
$record = [ordered]@{
    schema = 'learning_workflow_reproduction_v1'; smoke = [bool]$Smoke; python = $Python
    bc_manifest = $bcManifest; bc_training = $bcRun; bc_diagnostic_evaluation = $bcEvaluation
    label_manifest = $labelManifest; aggregated_training = $aggregatedRun; paired_evaluation = $pairedEvaluation
}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $pairedEvaluation 'reproduction.json') -Encoding utf8
Write-Host "Workflow complete. Paired results: $pairedEvaluation"
if ($Smoke) { Write-Host 'Smoke checks only; these outputs are not the documented 20-pair experiment.' }
