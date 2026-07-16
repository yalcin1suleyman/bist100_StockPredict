Get-ChildItem "c:\stajProje\*.csv" | ForEach-Object {
    $name = $_.Name
    $header = Get-Content $_.FullName -TotalCount 1
    $colCount = ($header -split ',').Count
    Write-Host "$name -> $colCount sutun"
    Write-Host "  Kolonlar: $header"
    Write-Host "---"
}
