# Starting the supervisor repository

The source-only archive `prototype-github-source.zip` contains the code, settings, tracks, tests and documentation. It has no Git history, training recordings or local environment. Extract it into a new folder, separate from the development checkout.

From that extracted `prototype/` folder:

```sh
git init -b main
git add .
git status --short
git commit -m "Initial driving prototype and population analysis"
```

Before committing, check that the staged files contain no training outputs, archives, environment folders or local secrets. The included `.gitignore` excludes these generated/local files, but keeps settings, track JSON assets and the analysis source.

Create an empty repository on GitHub, without an automatically generated README, then use its repository URL:

```sh
git remote add origin <repository-url>
git push -u origin main
```

Choose repository visibility and invite the supervisor through GitHub. A private repository can be shared by granting access. No repository was created or pushed automatically during this preparation.

The existing development checkout keeps its history and branches. Do not delete its `.git` folder or try to turn it into an initial commit. Use the exported source folder for that fresh start.

A fresh clone needs Python 3.11 or newer and a new virtual environment. Follow the main README to install dependencies, run tests and start training. Existing champions are shared separately because generated recordings and pickle genomes are not part of the source repository. Only load pickle genomes from a trusted source.
