$repo = "thanosprime/projectdavid-core"
$image = "projectdavid-core-api"
$now = Get-Date

# Get CI run finish times for main branch
$runs = gh run list --repo $repo --branch main --workflow "Lint, Test, Build, and Publish Docker Images" --limit 50 --json displayTitle,updatedAt,conclusion,headBranch | ConvertFrom-Json

# Get Docker Hub semver tags
$url = "https://hub.docker.com/v2/repositories/thanosprime/$image/tags/?page_size=100"
$allTags = @()
while ($url) {
    $resp = Invoke-RestMethod -Uri $url
    $allTags += $resp.results
    $url = $resp.next
}

$semverTags = $allTags | Where-Object { $_.name -match '^\d+\.\d+\.\d+$' }

# Match each tag to the closest CI run that finished before or around it
foreach ($tag in $semverTags | Sort-Object { [version]$_.name } -Descending) {
    $pulled = [datetime]$tag.tag_last_pulled
    $closestRun = $runs | Where-Object { $_.conclusion -eq "success" } |
        Sort-Object { [Math]::Abs(([datetime]$_.updatedAt - $pulled).TotalMinutes) } |
        Select-Object -First 1

    $ciFinished = if ($closestRun) { [datetime]$closestRun.updatedAt } else { $null }
    $gapHours = if ($ciFinished) { [math]::Round(($pulled - $ciFinished).TotalHours, 1) } else { "n/a" }
    $likely = if ($gapHours -is [double] -and $gapHours -gt 2) { "REAL USER" } else { "CI/YOU" }

    [PSCustomObject]@{
        Tag        = $tag.name
        CIFinished = if ($ciFinished) { $ciFinished.ToString("MM-dd HH:mm") } else { "n/a" }
        LastPulled = $pulled.ToString("MM-dd HH:mm")
        GapHours   = $gapHours
        Likely     = $likely
    }
} | Format-Table -AutoSize
