# BRD4 Predictor frontend (Next.js)

This is the Vercel half of the app. It accepts SMILES text or SDF/MOL2 files, shows all three pIC50 estimates and the three AD checks, and supports CSV export.

## Run locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` to the deployed FastAPI base URL, e.g. `https://brd4-backend.onrender.com` (no trailing slash). Deploy this folder as the frontend repository on Vercel and add the same variable in Vercel's Project Settings.
