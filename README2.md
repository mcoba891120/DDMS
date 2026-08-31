# CP revised version (for DAMASK3.0.0-alhpa4)
### Code
- [google drive](https://drive.google.com/drive/folders/1mcfi6g9-tCJwVlZYO-S-KCDgy6RWzmnB?usp=sharing)
- [github](https://github.com/KyleChien/crystal-plasticity/tree/master/ABAQUS_CPFEM)
### Usage
`python main.py [-i] YAML [-m] MODE [-tn] TASKNAME [-t] TARGET [-rd] ROOTDIR`
- YAML
   - name of the material parameters in `./configs/xxx.yaml`
- MDOE
   - choose modes, default: only show YS model results
   - `pre`: preprocessing, generating `xxx.vti` `material.yaml` `tensionX.load` files to `./ROOTDIR`
   - `post`: post processing, plot stress-strain curve in `./ROOTDIR`
- TASKNAME
   - your unique taskname, default: ''
- TARGET
   - target aging conditions, default: run all
   - e.g.
      - `7min_298K`
      - `7min` (i.e. `7min_298K` `7min_423K` `7min_473K` `7min_523K`)
      - `298K` (i.e. `7min_298K` `30min_298K` `6hr_298K` `168hr_298K`)
- ROOTDIR
   - root directory, default: `./cache/damask_results`

### brief memo
- mode `pre` uses `./configs/YAML` as template, will generate copies at `./ROOTDIR/TASKNAME`. so u can modify `./configs/YAML` file to ur desired settings and then run `main.py` in `pre` mode
- make sure u have `.hdf5` file under `./ROOTDIR/TASKNAME` before running `post` 
