$repo = "project-david-ai/projectdavid-core"
$image = "projectdavid-core-api"
$now = Get-Date

$runs = gh run list --repo $repo --branch main --workflow "Lint, Test, Build, and Publish Docker Images" --limit 50 --json displayTitle,updatedAt,conclusion,headBranch | ConvertFrom-Json | Where-Object { $_.displayTitle -match '^(fix|feat)' -and $_.conclusion -eq "success" }

$url = "https://hub.docker.com/v2/repositories/thanosprime/$image/tags/?page_size=100"
$allTags = @()
while ($url) {
    $resp = Invoke-RestMethod -Uri $url
    $allTags += $resp.results
    $url = $resp.next
}

$repoStats = Invoke-RestMethod -Uri "https://hub.docker.com/v2/repositories/thanosprime/$image/"
$semverTags = $allTags | Where-Object { $_.name -match '^\d+\.\d+\.\d+$' }
$results = @()

foreach ($tag in $semverTags | Sort-Object { [version]$_.name } -Descending) {
    $pulled = [datetime]$tag.tag_last_pulled
    $closestRun = $runs | Sort-Object { [Math]::Abs(([datetime]$_.updatedAt - $pulled).TotalMinutes) } | Select-Object -First 1
    $ciFinished = if ($closestRun) { [datetime]$closestRun.updatedAt } else { $null }
    $gapHours = if ($ciFinished) { [math]::Round(($pulled - $ciFinished).TotalHours, 1) } else { "n/a" }
    $likely = if ($gapHours -is [double] -and $gapHours -gt 2) { "REAL USER" } else { "CI/YOU" }
    $results += [PSCustomObject]@{ Tag = $tag.name; CIFinished = if ($ciFinished) { $ciFinished.ToString("MM-dd HH:mm") } else { "n/a" }; LastPulled = $pulled.ToString("MM-dd HH:mm"); PulledRaw = $pulled; GapHours = $gapHours; Likely = $likely }
}

$results | Select-Object Tag, CIFinished, LastPulled, GapHours, Likely | Format-Table -AutoSize

$realUsers = ($results | Where-Object { $_.Likely -eq "REAL USER" }).Count
$activeTags = ($results | Where-Object { ($now - $_.PulledRaw).TotalDays -lt 7 }).Count

Write-Host "--- Summary ---"
Write-Host ("Total pulls (all tags):    {0:N0}" -f $repoStats.pull_count)
Write-Host ("Confirmed real user tags:  {0}" -f $realUsers)
Write-Host ("Active tags (pulled <7d):  {0}" -f $activeTags)
