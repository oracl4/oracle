import time

last_time = int(round(time.time() * 1000))
while True:
	time_now = int(round(time.time() * 1000))
	if (time_now - last_time) > 5000:
		print("here")
		last_time = time_now