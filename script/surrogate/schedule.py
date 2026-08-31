# >-------------------------------------------------------
# > watch training data generation on Zinfandel
# > using ssh & sftp connections
# >-------------------------------------------------------

if __name__ == '__main__':
	import os
	import signal
	import argparse
	from getpass import getpass
	from ddms.surrogate.processing import RemoteWatcher, RepeatedTimer

	parser = argparse.ArgumentParser()
	parser.add_argument('--local-root', type=str, required=True, dest='local_root',
						 help='local directory to sync completed runs into')
	parser.add_argument('--remote-root', type=str, required=True, dest='remote_root',
						 help='remote directory containing DAMASK run folders')
	parser.add_argument('--remote-targets', nargs='+', required=True, dest='remote_targets',
						 help='subdirectories under --remote-root to watch, e.g. Mstiff Msmooth')
	args = parser.parse_args()

	watcher = RemoteWatcher(hostname=input('hostname: '),
							username=input('username: '),
							port=input('port: ') or '22',
							password=getpass('password: '))

	def func(**kwargs):
		if watcher.busy:
			print(f'watcher busying...')
			return

		local_root = args.local_root
		remote_root = args.remote_root
		remote_targets = args.remote_targets
		
		for remote_target in remote_targets:
			remote_dir = f'{remote_root}/{remote_target}'
			stdout = watcher.exec_command(
				f"""
				cd {remote_dir};
				
				for folder in ./morient* ;
				do 
					if [ -f "${{folder}}/${{folder}}.log" ] && grep -wq "terminated" "${{folder}}/${{folder}}.log"; 
					then 
						echo ${{folder}}
					fi 
				done 
				"""
			)[1]
			
			# download & remove file 
			for name in stdout.readlines():
				name = name.replace('\n', '')
				remote_path = f'{remote_dir}/{name}'
				local_path = f'{local_root}/{name}'	
				watcher.add_queue(remote_path, local_path)

		print('#'*50)
		watcher.exec_queue()

	timer = RepeatedTimer(5, func)
	timer.start()

	# ctrl+c event
	def exit_handler(sig, frame):
		timer.stop()
		watcher.finish()
	signal.signal(signal.SIGINT, exit_handler)