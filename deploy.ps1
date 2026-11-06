# Deploys the website/ folder to Vercel (production).
# Live URL: https://plant-disease-detection-dusky.vercel.app
#
# Uses the Vercel CLI login saved on this computer. If you prefer a token,
# save it in C:\Users\<you>\.vercel-token (outside the repo, never committed).

$site = Join-Path $PSScriptRoot "website"
$vercelArgs = @("deploy", $site, "--prod", "--yes", "--project", "plant-disease-detection")

$tokenFile = Join-Path $env:USERPROFILE ".vercel-token"
if (Test-Path $tokenFile) {
    $token = (Get-Content $tokenFile -Raw).Trim()
    if ($token) { $vercelArgs += @("--token", $token) }
}

npx --yes vercel@latest @vercelArgs
