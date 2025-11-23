# Deploy script to handle stuck git merge and push to main
Write-Host "Resolving git merge and deploying to main..."

# Set git editor to avoid vim issues
git config --global core.editor "notepad.exe"

# Navigate to project directory
Set-Location "c:\Users\jamin\OneDrive\NFL scrapping\NFL_DFS_OPTIMZER"

# Check if we're in the middle of a merge
if (Test-Path ".git/MERGE_HEAD") {
    Write-Host "Merge in progress detected, attempting to complete..."
    
    # Try to complete the merge with a commit message
    git commit -m "Merge feature/jamin-dev into main - ROI fixes and minimum point filter improvements"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Merge completed successfully!"
    } else {
        Write-Host "Merge failed, attempting to abort and retry..."
        git merge --abort
        
        # Try merge again
        git merge feature/jamin-dev -m "Merge feature/jamin-dev into main - ROI fixes and minimum point filter improvements"
    }
} else {
    Write-Host "No merge in progress, checking current status..."
    git status
}

# Push to main if we're on main branch
$currentBranch = git branch --show-current
if ($currentBranch -eq "main") {
    Write-Host "Pushing changes to main branch..."
    git push origin main
} else {
    Write-Host "Not on main branch, switching to main first..."
    git checkout main
    git merge feature/jamin-dev -m "Merge feature/jamin-dev into main - ROI fixes and minimum point filter improvements"
    git push origin main
}

Write-Host "Deployment complete!"