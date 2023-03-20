# TODO: If you wanted to go crazy:
# 	On the github gist that provided the minified help docs there are examples
# 	for displaying sections and using a template for coloring etc...
# 	Could go FULL MONO REPO and expose those at the root as well...

.PHONY: test
## Run related tests in test directory
test:
	poetry run pytest -rs --cov --cov-config=../pyproject.toml

.PHONY: test-file
## Run related tests in specific file
test-file:
	poetry run pytest -s ${file}

.PHONY: test-case
## Run related test case in specific file
test-case:
	poetry run pytest -s ${file} -k ${case}

.DEFAULT_GOAL := show-help
# See <https://gist.github.com/klmr/575726c7e05d8780505a> for explanation.
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more $(shell test $(shell uname) = "Darwin" && echo '-Xr')
