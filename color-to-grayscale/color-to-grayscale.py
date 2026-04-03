def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    if not image or not image[0]:
        return []
    h = len(image)
    w = len(image[0])
    gray = []
    for i in range(h):
        gray_row=[]
        for j in range(w):
            r, g, b = image[i][j]
            y = 0.299*r + 0.587*g + 0.114*b
            gray_row.append(y)

        gray.append(gray_row)

    return gray
            
            
        