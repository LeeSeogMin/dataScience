from pathlib import Path
save_path = Path(__file__).parent.parent.parent.parent / "diagram" / "test.txt"
save_path.write_text("testing pathing")
print(f"Created {save_path}")
