# TODO: Figure out a way to print help for ALL makefiles...
# @for file in $(MAKE_FILES); do $${file} ./tools/generate-makefile-help.sh; done
# @for file in $(MAKE_FILES); do echo $${file}; done

.PHONY: help
help:
	@FILE=Makefile ./tools/generate-makefile-help.sh
