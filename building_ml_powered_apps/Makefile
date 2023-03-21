# TODO: If you wanted to go crazy:
# 	On the github gist that provided the minified help docs there are examples
# 	for displaying sections and using a template for coloring etc...
# 	Could go FULL MONO REPO and expose those at the root as well...
TEST_BASE := poetry run pytest -rs
COVERAGE_OPTS := --cov --cov-config=../pyproject.toml

.PHONY: test
## Run related tests in test directory
test:
	$(TEST_BASE) $(COVERAGE_OPTS)

.PHONY: test-file
## Run related tests in specific file
test-file:
	$(TEST_BASE) ${file}

.PHONY: test-case
## Run related test case in specific file
test-case:
	$(TEST_BASE) ${file} -k ${case}

.PHONY: html-cover
## Run tests in test directory and create html coverage and open
## for viewing
html-cover:
	$(TEST_BASE) $(COVERAGE_OPTS) --cov-report=html
	firefox cov_html/index.html

.DEFAULT_GOAL := show-help
# See <https://gist.github.com/klmr/575726c7e05d8780505a> for explanation.
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more $(shell test $(shell uname) = "Darwin" && echo '-Xr')
