file bread_debug.exe
set args --server --tokens 100
set pagination off
set print pretty on

# Run the program
run < /tmp/test_single_query.txt &

# Wait a bit for it to load
shell sleep 30

# Attach to the process if needed
# For now, let's just run to completion
continue

quit
