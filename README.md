# DiscourseMarkers

source venv/bin/activate

nvidia-smi

## Screen commands
Create: screen -S *screen_name*

Leave: Ctrl+a -> d

Return: screen -r *screen_name*

Delete:
- inside the screen: exit 
- outside the screen: screen -X -S *screen_name* quit

### Run 
python main.py --mode transfer --step 0 > main_transfer_0.txt 2>&1 &
tail -f main_transfer_0.txt