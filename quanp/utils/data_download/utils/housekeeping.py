import os, sys

# 1: Checks if in proper number of arguments are passed gives instructions on proper use.
def argsCheck_main(numArgs):
	if len(sys.argv) < numArgs:
		print('To start the program, please insert 1 or more tickers separated by comma')
		print('Usage: {} ListOfTickers'.format(sys.argv[0]))
		print('Examples: {} AAPL,AMZN,CHWY,AAL,DAL'.format(sys.argv[0]))
		exit(1) # Aborts program. (exit(1) indicates that an error occurred)

# 2: delete files generated in the previous analyses.
def delete_files(pth):
    for sub in os.listdir(pth):
        if os.path.isdir(pth + '/' + sub):
            None
        else:
            os.remove(pth + '/' + sub)
