# Prompt pour générer une vidéo promotionnelle d'AIPROD en 10 secondes

$json = @{
    content      = @"
AIPROD: Transform Text to Professional Video in 4K

Opening: [Dark background with subtle motion] 
Narrator: "Script to 4K video in seconds"

[Scene 1 - 3 sec] Fast transitions of text transforming into vibrant video frames
Show: Script document → stunning video scene
Colors: Pink, blue, yellow gradients
Text overlay: "Transform Scripts to Video"

[Scene 2 - 2 sec] Show icons: AI chip, lightning bolt, camera
Emphasize: AI-powered Intelligence
Speed indicator: "10x faster"

[Scene 3 - 2 sec] Professional video clips showing quality
Text: "Enterprise-Grade Quality"
Subtitle: "4K Resolution, 60fps"

[Scene 4 - 3 sec] Finale: 
Logo appears (AIPROD film reel)
Call to action: "Visit GitHub for Code"
Color fade: Rainbow to AIPROD brand colors

Overall tone: Professional, dynamic, innovative
Music: Upbeat, modern, tech-forward
Duration: 10 seconds total
"@
    duration_sec = 10
    preset       = "brand_campaign"
    priority     = "high"
    lang         = "en"
} | ConvertTo-Json -Depth 10

$headers = @{
    "Content-Type" = "application/json"
}

Write-Host "🎬 Generating promotional video via AIPROD API..."
Write-Host "Sending POST request to http://localhost:8000/pipeline/run"
Write-Host "Prompt length: $($json.Length) characters"
Write-Host ""

try {
    $response = Invoke-WebRequest `
        -Uri "http://localhost:8000/pipeline/run" `
        -Method POST `
        -Headers $headers `
        -Body $json `
        -UseBasicParsing `
        -ErrorAction Stop

    $result = $response.Content | ConvertFrom-Json
    Write-Host "✅ Success! Response received:"
    Write-Host ($result | ConvertTo-Json -Depth 5)
    
    # Extract job_id if available
    if ($result.data.job_id) {
        Write-Host ""
        Write-Host "📋 Job ID: $($result.data.job_id)"
        Write-Host "Check status: /pipeline/job/$($result.data.job_id)"
    }
}
catch {
    Write-Host "❌ Error: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        Write-Host "Status: $($_.Exception.Response.StatusCode)"
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        Write-Host "Response: $($reader.ReadToEnd())"
    }
}
