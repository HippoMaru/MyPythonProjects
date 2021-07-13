from PIL import Image


# frames = [Image.open(f'images/homer-{i}.jpg') for i in range(1, 11)]
# frames[0].save(
#     'homer.gif',
#     save_all=True,
#     append_images=frames[1:],
#     optimize=True,
#     duration=100,
#     loop=0
# )

frames = [Image.open(f'images/sad-{i}.jpg') for i in range(11)]
frames[0].save(
    'sad.gif',
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=[100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 400],
    loop=0
)
