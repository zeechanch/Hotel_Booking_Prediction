# Hotel Booking Prediction frontend

An Angular frontend for the hotel booking cancellation research project.

## Run locally

```bash
npm ci
npm start
```

## Production build

```bash
npm run build
```

The GitHub Actions workflow in `../.github/workflows/angular-pages.yml` runs the
unit tests and production build on pull requests and pushes to `main`. It deploys
successful builds from `main` to GitHub Pages. Enable **Settings → Pages → Build
and deployment → GitHub Actions** in the GitHub repository once before deploying.
