# Publishing to PyPI

This project is configured to automatically publish to PyPI when a new GitHub release is created.

## Setting up PyPI Token

To enable automatic publishing, you need to set up a PyPI API token as a GitHub secret:

1. **Create a PyPI API token**:
   - Log in to your PyPI account at https://pypi.org/
   - Go to Account Settings → API tokens
   - Create a new API token with scope "Entire account (all projects)"
   - Copy the token value (you won't be able to see it again)

2. **Add the token to GitHub secrets**:
   - Go to your GitHub repository
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste your PyPI token
   - Click "Add secret"

## Creating a Release

To trigger the publishing workflow:

1. Go to your GitHub repository
2. Navigate to Releases
3. Click "Create a new release"
4. Choose a tag version (e.g., v0.1.3)
5. Add a title and description
6. Click "Publish release"

The GitHub Actions workflow will automatically build and publish your package to PyPI.

## Manual Publishing

If you need to publish manually, you can use `uv` for faster builds and publishing:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build the package
uv build

# Upload to PyPI
uv publish --username __token__ --password YOUR_PYPI_TOKEN
```

Alternatively, you can use the standard Python tools:

```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```
