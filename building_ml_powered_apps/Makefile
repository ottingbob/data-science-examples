BUILDING_ML_POWERED_APPS_PROJECT_NAME := building_ml_powered_apps
PROJECT_NAMES += $(BUILDING_ML_POWERED_APPS_PROJECT_NAME)

TEST_BASE := poetry run pytest $(BUILDING_ML_POWERED_APPS_PATH) -rs
COVERAGE_OPTS := --cov --cov-config=./pyproject.toml

## @section General

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).help
## Displays help for the current project.
## This is the default target
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).help:
	@FILE=$(BUILDING_ML_POWERED_APPS_PATH)/Makefile \
		PROJECT_NAME=$(BUILDING_ML_POWERED_APPS_PROJECT_NAME) \
		./tools/generate-makefile-help.sh

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME)
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME): $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).help

## @section Application

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).run
## This runs a demo Flask app to serve the ML models for question improvement
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).run:
		FLASK_APP=$(BUILDING_ML_POWERED_APPS_PATH)/webapp.py poetry run flask run

## @section Testing

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test
## Run related tests in test directory
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test:
	$(TEST_BASE) $(COVERAGE_OPTS)

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test-file
## Run related tests in specific file
## @param file=building_ml_powered_apps/tests/test_model.py File to test
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test-file:
	$(TEST_BASE) ${file}

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test-case
## Run related test case in specific file
## @param file=building_ml_powered_apps/tests/test_model.py File to test
## @param case=test_model_proba Test case regex from file to run
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).test-case:
	$(TEST_BASE) ${file} -k ${case}

.PHONY: $(BUILDING_ML_POWERED_APPS_PROJECT_NAME).html-cover
## Run tests in test directory and create html coverage and open
## for viewing
$(BUILDING_ML_POWERED_APPS_PROJECT_NAME).html-cover:
	$(TEST_BASE) $(COVERAGE_OPTS) --cov-report=html
	firefox $(BUILDING_ML_POWERED_APPS_PATH)/cov_html/index.html

# FIXME: This has been adapted to the top-level Makefile setup to scale
# 	to other projects...
# See <https://gist.github.com/klmr/575726c7e05d8780505a> for explanation.
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more $(shell test $(shell uname) = "Darwin" && echo '-Xr')
