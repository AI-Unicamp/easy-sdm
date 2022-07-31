project-packages=easy_sdm
project-name=easy_sdm
project-minimum-coverage=65.0
time-stamp=$(shell date"+%Y-%m-%d-%H%M%S")
WORKSPACE_TMP=reports
DOCKER_IMAGE_NAME=${project-name}
DOCKERFILE=dockerfile
ENVIRON=production

.DEFAULT_GOAL = help
.PHONY = help
help:
	@echo "available commands"
	@echo " - install     : installs all requirements"
	@echo " - dev         : installs all development requirements"
	@echo " - test-code   : run all unit tests"
	@echo " - clean       : cleans up all folders"
	@echo " - flake       : runs flake8 style checks"
	@echo "Other functionalities check Makefile targets"

install:
	poetry install --no-dev

install-full:
	poetry install

dev: install-full
	poetry run pre-commit install

s2i-mac-install:
	brew install source-to-image

.PHONY = test-code
test-code:
	pytest --disable-pytest-warnings --junit-xml ${WORKSPACE_TMP}/unit_tests.xml

.PHONY = hadolint
hadolint:
	docker run --rm -i hadolint/hadolint < ${DOCKERFILE}

.PHONY = flake
flake:
	flake8 .

.PHONY = black
black:
	black ${project-packages}

.PHONY = isort
isort:
	isort ${project-packages}

.PHONY = code-complexity
code-complexity:
	radon mi --sort --json --output-file ${WORKSPACE_TMP}/mi_report.json .
	radon cc --json --total-average --output-file ${WORKSPACE_TMP}/cc_report.json .
	xenon --max-absolute B --max-modules B --max-average A .

.PHONY = codefresh-test
codefresh-test:
	pytest --disable-pytest-warnings --cov-report=xml:coverage.xml --cov=${project-packages} --cov-fail-under=${project-minimum-coverage}

.PHONY = test-coverage
test-coverage:
	poetry run pytest --disable-pytest-warnings --cov-report=xml:coverage.xml --cov-report html:cov_html --cov=${project-packages} --cov-fail-under=${project-minimum-coverage}

.PHONY = clean
clean:
	rm -rf coverage .coverage*
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '.project' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	rm -rf dist

coverage: ${WORKSPACE_TMP} test-coverage

static-analysis: ${WORKSPACE_TMP} code-complexity flake

patch:
	poetry run bumpver update --patch --commit-message "[patch-version] {old_version} -> {new_version}"

minor:
	poetry run bumpver update --minor --commit-message "[minor-version] {old_version} -> {new_version}"

major:
	poetry run bumpver update --major --commit-message "[major-version] {old_version} -> {new_version}"

cli-docs:
	typer hermod.console utils docs --output README-cli.md --name hermod

#Docker
docker-build:
	docker build --progress plain --build-arg ENVIRON=${ENVIRON} -f ${DOCKERFILE} -t ${DOCKER_IMAGE_NAME} .

test-builder-image:
	IMAGE_NAME=${DOCKER_IMAGE_NAME} tests/s2i/run

builder-usage:
	s2i usage ${DOCKER_IMAGE_NAME}

# git
commit: black
	git add .
	PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "$t" -m "$b"
	git push
