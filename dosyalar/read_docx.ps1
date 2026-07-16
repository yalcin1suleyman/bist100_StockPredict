param([string]$FilePath)

Add-Type -AssemblyName 'System.IO.Compression.FileSystem'
$zip = [System.IO.Compression.ZipFile]::OpenRead($FilePath)
$entry = $zip.GetEntry('word/document.xml')
$stream = $entry.Open()
$reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
$content = $reader.ReadToEnd()
$reader.Close()
$stream.Close()
$zip.Dispose()

$text = [regex]::Replace($content, '<[^>]+>', "`n")
$text = $text -replace '^\s*$\n', '' -replace '\n{3,}', "`n`n"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Output $text.Trim()
