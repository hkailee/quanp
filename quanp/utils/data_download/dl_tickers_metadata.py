#!/Users/leehongkai/anaconda/envs/trading/bin/python
import os, sys
from utils.company_metadata import retrieve_wiki_sp500_tickers

# 1: Checks if in proper number of arguments are passed gives instructions on proper use.
def argsCheck(numArgs):
	if len(sys.argv) < numArgs:
		print('To start the program, please provide an', 
				'output directory (absolute or relative path)')
		print('Usage: {} [Output directory]'.format(sys.argv[0]))
		print('Examples: {} Data'.format(sys.argv[0]))
		exit(1) # Aborts program. (exit(1) indicates that an error occurred)

argsCheck(2)

# 2. the output directory
output_dir = sys.argv[1]

# 3. create output directory if not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if __name__ == '__main__':
    retrieve_wiki_sp500_tickers(output_dir)
