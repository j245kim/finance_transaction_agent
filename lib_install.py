import subprocess
import pkg_resources

packges = [dist.project_name for dist in pkg_resources.working_set]

for package in packges:
    subprocess.run(['pip', 'install', '--upgrade', package])
    