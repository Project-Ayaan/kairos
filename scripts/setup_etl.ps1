# Setup dependencies for Kairos ETL and Streamlit UI
Write-Host "Syncing uv dependency groups: etl, ui..."
uv sync --group etl --group ui
Write-Host "Setup completed successfully!"
