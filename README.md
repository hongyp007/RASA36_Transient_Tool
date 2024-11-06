# RASA36 Transient Tool

## Purpose

To identify transient candidates in RASA36 science images using a GUI tool.

The tool provides a graphical interface for examining and classifying potential transient astronomical objects detected in RASA36 telescope images. It displays three views:

1. A subtracted image showing the difference between science and reference images to highlight changes
2. The new science image showing the current observation
3. A reference image from a previous epoch for comparison

### Key features:
- Interactive zoom and pan controls for detailed inspection
- Side-by-side display of science, reference and difference images
- Quick classification using keyboard shortcuts
- Progress tracking across multiple image tiles
- Memo field for notes on individual candidates
- Flexible configuration via config.ini
- Automatic image scaling and normalization
- Image caching for smooth navigation
- CSV output for analysis and followup

The tool is designed to efficiently process large numbers of candidate detections while maintaining high accuracy through careful visual inspection and comparison.


## Requirements 

The image format is either PNG or FITS.
Python 3.10 or later is required.

### Required Python packages:
- numpy
- pandas 
- astropy
- matplotlib
- tkinter


## Usage

1. Install required dependencies:
   ```bash
   pip install numpy pandas astropy matplotlib tkinter
   ```

2. Configure settings in `config.ini`:
   - Set data directory path containing FITS/PNG images
   - Configure display preferences (zoom, scaling, etc.)
   - Set keyboard shortcuts
   - Choose operation mode (normal/view-only)
   - Adjust cache and preload settings

3. Run the tool:
   ```bash
   python3 Transient_Tool_RASA36.py
   ```

4. Interface Features:
   - Image display with zoom/pan controls
   - Toggle between science and reference images
   - Classification buttons for candidates:
     - Significant (Q)
     - Marginal (W) 
     - Subtraction artifact (E)
     - Error (R)
     - You can change the name of the categories in `config.ini`
   - Navigation controls:
     - Next/Previous image (Arrow keys)
     - Jump to unclassified (U)
     - Reset zoom (T)
   - Results saved to CSV file
   - Short-cut keys can be changed in `config.ini`

5. Output:
   - Classifications stored in CSV with columns:
     - File index
     - Tile ID
     - Unique number
     - Display scale
     - Memo field
     - Classification (Significant, Marginal, Subtraction artifact, Error)

6. Modes:
   - Normal mode: Full classification capabilities
   - View mode: Display-only without classification
   - Specific view mode: Filter by classification type

7. Logging:
   - Operations logged to configured log file
   - Configurable log level (DEBUG/INFO/WARNING/ERROR)

For detailed configuration options, refer to comments in `config.ini`.

## Now working on

- Making another tool to make visualization of transient candidates

## Future Work

- Make the code more efficient
- Generalize the code for other telescopes
- Add more analysis tools
