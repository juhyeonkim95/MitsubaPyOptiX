import cv2
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont


def crop_image(img, lower=True):
	# Create a mask image with a triangle on it
	y, x, _ = img.shape
	mask = np.zeros((y, x), np.uint8)
	if lower:
		triangle_cnt = np.array([(x, y), (x, 0), (0, y)])
	else:
		triangle_cnt = np.array([(0, 0), (0, y), (x, 0)])
	cv2.drawContours(mask, [triangle_cnt], 0, 255, -1)
	img = cv2.bitwise_and(img, img, mask=mask)
	return img


def pil_to_opencv(pil_image):
	numpy_image = np.array(pil_image)
	opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
	return opencv_image


def opencv_to_pil(opencv_image):
	color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
	pil_image = Image.fromarray(color_converted)
	return pil_image


def crop_image_center(image, w, h):
	w = int(w)
	h = int(h)
	print("H", h, "W", w)
	x = (image.shape[1] - w) // 2
	y = (image.shape[0] - h) // 2
	crop_img = image[y:y + h, x:x + w]
	return crop_img


def write_msg(img, msg, bottom=False, margin=5):
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 36)
	# draw.textsize(msg, font=font)

	w, h = draw.textsize(msg, font=font)
	W, H = img.size
	if not bottom:
		pos = margin
	else:
		pos = H - h - margin
	draw.text(((W - w) / 2, pos), msg, fill="white", font=font)


def write_msg2(img, msg, margin=5):
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

	w, h = draw.textsize(msg, font=font)
	W, H = img.size
	draw.text((margin, margin), msg, fill="white", font=font)


def merge_video(scene_name, scale=3, crop=True):
	cap_rl = cv2.VideoCapture('videos/%s/path_tracing_with_rl.avi' % scene_name)
	cap_no_rl = cv2.VideoCapture('videos/%s/path_tracing_without_rl.avi' % scene_name)
	frame_count = 0
	img_array = []
	height = 0
	width = 0
	size = (0, 0)
	new_size = (0,0)
	while True:
		frame_count += 1
		ret, frame_rl = cap_rl.read()
		ret2, frame_no_rl = cap_no_rl.read()

		if not ret:
			break
		size = frame_rl.shape[0:2]
		width = size[1]
		height = size[0]
		if crop:
			frame_rl = crop_image_center(frame_rl, width * 0.75, height)
			frame_no_rl = crop_image_center(frame_no_rl, width * 0.75, height)
			size = frame_rl.shape[0:2]
		rl_bottom=False
		frame_rl = crop_image(frame_rl, rl_bottom)
		frame_no_rl = crop_image(frame_no_rl, not rl_bottom)
		frame = frame_rl+frame_no_rl
		n_width = size[1] // scale
		n_height = size[0] // scale
		new_size = (n_width, n_height)
		frame = cv2.resize(frame, dsize=new_size)
		pil_image = opencv_to_pil(frame)
		write_msg(pil_image, "with path guiding", bottom=rl_bottom)
		write_msg(pil_image, "without path guiding", bottom=not rl_bottom)
		write_msg2(pil_image, "SPP : %d" % (frame_count * 8))

		frame = pil_to_opencv(pil_image)
		img_array.append(frame)
	cap_no_rl.release()
	cap_rl.release()

	# out = cv2.VideoWriter('videos/%s/merged.avi' % scene_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, new_size)
	#
	# for i in range(len(img_array)):
	# 	img = img_array[i]
	# 	out.write(img)
	# out.release()
	# cv2.destroyAllWindows()

	print("Saving GIF file")
	with imageio.get_writer("videos/%s/merged.gif" % scene_name, mode="I") as writer:
		for idx, frame in enumerate(img_array):
			print("Adding frame to GIF file: ", idx + 1)
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			writer.append_data(rgb_frame)


if __name__ == "__main__":
	merge_video("veach_door_simple", scale=3)
