import argparse

parser = argparse.ArgumentParser(description='Basic app to get summaries for the blogs listed at jobleads.com/career-advice')
parser.add_argument('refetch_blogs',type=bool, default=False, help= 'Set this flag to True to force scrape the blogs from the website')
parser.add_argument('envfile_path', type=str, default='./envfile', help='Path to find the env file containing any required keys or secrets')

args = parser.parse_args()

print('This is a python app to summarize the blog posts posted at jobleads.com/career-advice.')
print('Select an option...')
print('1. List blog titles.')
print('2. Refetch blog titles.')

