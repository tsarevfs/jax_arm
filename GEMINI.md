## Source control
Use jj.
Keeep commit messages short. First line should not have any prefix. Only use word "refactoring" when code have no functional changes. 

## Comments

- Don't comment obvious. Don't repeat code logic in comments.
- Prefer writing structured code (e.g. extracting method with good name) rather than add a comment.

## Running python

- Try to reuse virtual environment in .env
- You don't have interactive shell, so combine commands, e.g. `source .env/bin/activate; python3 -m unittest tests/test_mjx_env.py`