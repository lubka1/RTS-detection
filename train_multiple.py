import subprocess

configs = [
    ["python", "train.py", "--fusion", "early"],
    ["python", "train.py", "--fusion", "middle", "--strategy", "concat"],
    ["python", "train.py", "--fusion", "middle", "--strategy", "average"],
    ["python", "train.py", "--fusion", "middle", "--strategy", "concat", "--attention"],
    ["python", "train.py", "--fusion", "middle", "--strategy", "average", "--attention"],
    ["python", "train.py", "--fusion", "late", "--strategy", "concat"],
    ["python", "train.py", "--fusion", "late", "--strategy", "average"],
    ["python", "train.py", "--fusion", "late", "--strategy", "concat", "--attention"],
    ["python", "train.py", "--fusion", "late", "--strategy", "average", "--attention"],
]

for cmd in configs:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
