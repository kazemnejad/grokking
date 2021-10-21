import argparse
import json
import shlex
import site
import subprocess
import tempfile
from pathlib import Path

from comet_ml import Artifact, Experiment

site.addsitedir("src/")
from common import py_utils
from common.config import HoconConfig

cmd_guide = """
python3 scripts/run_experiment.py --platform mila --slurm-args '"--gres=gpu:1"' \\
            --image deepl-tf_v0.1.sif --assets "*" {bundle}   
"""


def make_run_script(configs: str, commands: str, env_vars: str, exp_key: str) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script += f"export COMET_EXPERIMENT_KEY={exp_key}\n"

    script += (
        "\n\nexport PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages/:$PYTHONPATH\n"
    )
    script += "\n\npip install --user -r src/requirements.txt\n"

    configs_str = configs
    script += "\n\n"
    for c in commands.split(","):
        c = c.strip()
        script += f"python src/main.py --configs '{configs_str}'\\\n"
        script += f"       {c}\n\n"

    script += 'echo "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    subprocess.check_call(shlex.split(f"nano {script_path}"))

    return script_path


def make_metadata(exp_name, exp_key):
    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tmp_dir / "metadata.json"

    metadata = {"exp_name": exp_name, "exp_key": exp_key}

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path


def get_exp_name(configs):
    filenames = list(map(lambda x: x.strip(), configs.split(",")))
    config = HoconConfig.from_files(filenames)

    config.put("config_filenames", filenames)

    exp_name = py_utils.unique_experiment_name(config)

    return exp_name


def upload_script(script_path, metadata_path, exp_name):
    cmd_bundle_name = f"scripts-{exp_name}"
    subprocess.check_call(
        shlex.split(
            f'cl upload {script_path} {metadata_path} -n "{cmd_bundle_name}" --tags script'
        )
    )

    return cmd_bundle_name


def upload_src():
    subprocess.check_call(
        shlex.split(f'cl upload src -n src -x "__pycache__" --tags src')
    )


def upload_scripts():
    subprocess.check_call(
        shlex.split(f'cl upload scripts -n scripts -x "__pycache__" --tags scripts')
    )


def make_experiment_bundle(cmd_bundle_name, dataset_name, exp_name):
    output = subprocess.check_output(
        shlex.split(
            f"cl make run.sh:{cmd_bundle_name}/run.sh "
            + f"metadata.json:{cmd_bundle_name}/metadata.json "
            + f'data:{dataset_name} src:src configs:configs scripts:scripts -n "exp-{exp_name}" --tags exp'
        )
    )

    bundle_name = output.decode("utf8").split("\n")[0]
    return bundle_name


def set_worksheet(codalab_ws):
    subprocess.check_call(shlex.split(f"cl work {codalab_ws}"))


def upload_configs():
    subprocess.check_call(shlex.split(f"cl upload configs -n configs --tags configs"))


def main2(
        package: str,
        configs: str,
        dataset_name: str,
        commands: str,
        env_vars: str,
        codalab_ws: str = "kazemnejad-comp-gen",
):
    script_path = make_run_script(package, configs, commands, env_vars)
    print("# ----> 1. Generating a unique experiment name...")
    exp_name = get_exp_name(package, configs)
    print(f"Experiment Name: {exp_name}")

    metadata_path = make_metadata(package, configs, exp_name, commands, env_vars)

    print(f"# ----> 2. Setting current worksheet to {codalab_ws}")
    set_worksheet(codalab_ws)
    print(f"# ----> 3. Uploading run script..")
    cmd_bundle_name = upload_script(script_path, metadata_path, exp_name)
    print(f"# ----> 4. Uploading the current source code...")
    upload_src()
    upload_scripts()
    print(f"# ----> 5. Uploading the current configs...")
    upload_configs()
    print(f"# ----> 6. Creating the experiment bundle...")
    bundle_name = make_experiment_bundle(cmd_bundle_name, dataset_name, exp_name)

    print(f"\n\nBundle ID: {bundle_name}")
    print(f"URL: 'https://worksheets.codalab.org/bundles/{bundle_name}'")
    print(f"Exp Name: {exp_name}\n")
    print(f"# ------>  Done! Run the following command to run the experiment")
    print(cmd_guide.format(bundle=bundle_name))

def add_folder(artifact:Artifact, path: Path, logic_path: str):
    org_path = str(path)
    for i in path.glob('**/*'):
        if i.name == "__pycache__" or i.name.endswith(".pyc") or i.is_dir():
            continue

        child_path = str(i)[len(org_path)+1:]
        artifact.add(i, f"{logic_path}/{child_path}")

def main(args: argparse.Namespace):
    project: str = args.project
    configs: str = args.configs

    print("# ----> 1. Generating a unique experiment name...")
    exp_name = get_exp_name(configs)

    if args.dataset:
        ds_name = args.dataset.replace("/", "_")
        exp_name += f"___{ds_name}"

    exp = Experiment(
        project_name=project,
        log_code=False,
        log_graph=False,
        log_env_cpu=False,
        log_env_gpu=False,
        log_git_patch=False,
        log_env_details=False,
        log_git_metadata=False,
        log_env_host=False,
        parse_args=False,
    )
    exp.set_name(exp_name)
    exp_key = exp.get_key()

    run_script_path = make_run_script(configs, args.commands, args.env_vars, exp_key)
    metadata_path = make_metadata(exp_name, exp_key)

    artifact_name = f"artf-{exp_key}"
    artifact = Artifact(name=artifact_name, artifact_type="experiment_bundle")
    add_folder(artifact, Path("configs"), "configs")
    add_folder(artifact, Path("src"), "src")
    artifact.add(str(run_script_path), logical_path="run.sh")
    artifact.add(str(metadata_path), logical_path="metadata.json")
    if args.dataset:
        artifact.add_remote(args.dataset, "data")

    exp.log_artifact(artifact)
    exp.end()

    print(f"\n\nExp Key: {exp_key}")
    print(f"Exp URL: {exp.url}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    parser.add_argument(
        "-s",
        "--configs",
        metavar="CONFIGS[,CONFIGS,CONFIGS]",
        type=str,
        help="Config file names",
    )

    parser.add_argument(
        "-c",
        "--commands",
        metavar="cmd -a -b[,cmd -c -d]",
        type=str,
        help="Experiment commands",
    )

    parser.add_argument(
        "-d", "--dataset", metavar="DATASET", type=str, help="Dataset name's bundle"
    )

    parser.add_argument(
        "-p",
        "--project",
        metavar="project",
        type=str,
        default="grokking",
        help="CometML project",
    )

    parser.add_argument(
        "-e",
        "--env-vars",
        metavar="KEY=VAL[,KEY=VAL]",
        type=str,
        help="Experiment environment variables",
    )

    args = parser.parse_args()

    main(args)
