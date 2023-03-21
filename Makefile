# Include colours
YELLOW_COLOUR := '\033[33m%s\033[0m\n'
BLUE_COLOUR := '\033[36m%s\033[0m\n'

# Include projects
PROJECTS_PATH = .
BUILDING_ML_POWERED_APPS_PATH = $(PROJECTS_PATH)/building_ml_powered_apps
BUILDING_ML_POWERED_APPS_MAKE_FILE = $(BUILDING_ML_POWERED_APPS_PATH)/Makefile
include $(BUILDING_ML_POWERED_APPS_MAKE_FILE)

# FIXME: Currently I can't iterate over the `MAKE_FILES` param so
# 	this is not being used right now...
MAKE_FILES = Makefile
MAKE_FILES += $(BUILDING_ML_POWERED_APPS_MAKE_FILE)

# Generate the `help` target from Makefile comments
include ./tools/Makefile-help.mk

## @section Projects

.PHONY: list
## List the projects defined with the make setup
list:
	@printf ${YELLOW_COLOUR} "Available Projects"
	@printf -- '%*s\n' "30" | tr ' ' "-"
	@for p in $(PROJECT_NAMES); do printf ${BLUE_COLOUR} "$${p}"; done

# Explicitly define the default goal at top level
.DEFAULT_GOAL := help
