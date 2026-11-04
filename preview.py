"""Render a paired comparison of fixed-coefficient receiver responses."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_preview(path, responses, pairs, frequency, time):
    """Use one magnitude scale for both waveforms and all three columns."""
    images = []
    for pair in pairs:
        base = responses[pair['background_case']]
        echo = responses[pair['echo_case']]
        images.append((abs(base), abs(echo), abs(echo - base)))
    scale = max(float(a.max()) for row in images for a in row)
    colors = np.array([[17, 38, 55], [36, 104, 131], [69, 166, 164],
                       [213, 229, 198], [247, 190, 102]], dtype=float)
    levels = np.linspace(0, scale if scale else 1, len(colors))
    canvas = Image.new('RGB', (1030, 573), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    navy, muted = '#163449', '#426174'
    heading = ImageFont.load_default(size=26)
    label = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    draw.text((20, 14), 'PITM | Paired synthetic receiver responses', fill=navy, font=heading)
    draw.text((20, 53), 'Each row uses the same background realization and reference for both inputs.',
              fill=muted, font=label)
    column_x = (92, 409, 726)
    titles = ('Background only', 'Background + echo', 'Absolute paired change')
    for x, title in zip(column_x, titles):
        draw.text((x, 89), title, fill=navy, font=label)
    width, height = 288, 140
    for row_index, (pair, row) in enumerate(zip(pairs, images)):
        y = 125 + row_index * 192
        draw.text((12, y + 48), f"{pair['center_hz']:g} Hz", fill=navy, font=label)
        draw.text((12, y + 73), f"{pair['width_s']:g} s", fill=navy, font=label)
        for x, field in zip(column_x, row):
            rgb = np.stack([np.interp(field[::-1], levels, colors[:, i])
                            for i in range(3)], axis=-1).astype('uint8')
            tile = Image.fromarray(rgb).resize((width, height), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y))
            draw.text((x, y + height + 4), f'{time[0]:g} s', fill=muted, font=small)
            draw.text((x + width, y + height + 4), f'{time[-1]:g} s',
                      fill=muted, font=small, anchor='ra')
    draw.text((20, 501), f'Vertical axis: frequency, {frequency[0]:g} Hz (bottom) to '
              f'{frequency[-1]:g} Hz (top). Horizontal axis: time.', fill=muted, font=label)
    gradient = np.stack([np.interp(np.linspace(0, levels[-1], 152), levels, colors[:, i])
                         for i in range(3)], axis=-1).astype('uint8')
    bar = Image.fromarray(gradient[None, ...]).resize((152, 13))
    canvas.paste(bar, (21, 542))
    draw.text((183, 536), f'Shared scale: 0 to {scale:.3f} arbitrary response units',
              fill=navy, font=label)
    canvas.save(path)
