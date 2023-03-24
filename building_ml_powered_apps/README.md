### Building ML Powered Apps

```bash
$ make building_ml_powered_apps
Usage: make [TARGET [TARGET ...]] [ARGUMENT=VALUE [ARGUMENT=VALUE ...]]
General:
    building_ml_powered_apps.help:
      Displays help for the current project.
      This is the default target

Application:
    building_ml_powered_apps.run:
      This runs a demo Flask app to serve the ML models for question improvement

Testing:
    building_ml_powered_apps.test:
      Run related tests in test directory

    building_ml_powered_apps.test-file: file=building_ml_powered_apps/tests/test_model.py
      Run related tests in specific file

      Params:
       - file (ex: building_ml_powered_apps/tests/test_model.py) File to test

    building_ml_powered_apps.test-case: file=building_ml_powered_apps/tests/test_model.py case=test_model_proba
      Run related test case in specific file

      Params:
       - file (ex: building_ml_powered_apps/tests/test_model.py) File to test
       - case (ex: test_model_proba) Test case regex from file to run

    building_ml_powered_apps.html-cover:
      Run tests in test directory and create html coverage and open for viewing
```
