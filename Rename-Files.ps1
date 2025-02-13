function Rename-Files {
    param (
        [string]$Path
    )

    # Get all files and directories in the specified path
    $items = Get-ChildItem -Path $Path -Recurse

    foreach ($item in $items) {
        if ($item -is [System.IO.FileInfo]) {
            # Check if the file name starts with "DiffSharp."
            if ($item.Name -like "DiffSharp-*") {
                # Construct the new file name
                $newName = $item.Name -replace "^DiffSharp\-", "Furnace-"
                # Rename the file
                Rename-Item -Path $item.FullName -NewName $newName
            }
        }
    }
}

# Call the function with the current directory
Rename-Files -Path (Get-Location)