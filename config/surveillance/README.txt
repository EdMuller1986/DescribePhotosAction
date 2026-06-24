Private camera files are NOT stored in this public repository.

First-time setup on Google Drive:

1. Upload a daytime reference photo to your camera folder root
   (for example reference_day.jpg or any JPG/PNG snapshot).

2. Run GitHub Actions workflow "Calibrate surveillance from Google Drive"
   with the folder URL. It writes:

     config/surveillance.json
     config/ignore_mask.png
     config/reference_day.jpg
     config/zones_preview.jpg

3. Upload daily videos (.mkv) to the same folder root.

4. Run "Surveillance day summary from Google Drive" with the folder URL.
   All videos without an existing .summary.json are processed.

Optional env overrides (repository variables):

  SURVEILLANCE_REFERENCE_FILE   explicit reference image path on Drive
  SURVEILLANCE_FORCE_REPROCESS    true to rebuild summaries
  SURVEILLANCE_MAX_VIDEOS_PER_RUN limit videos per run (0 = no limit)