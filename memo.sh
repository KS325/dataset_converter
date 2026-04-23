
nohup bash -c '
/home/sato/dev/dataset_converter/.venv_n/bin/python main.py --data_name "KS325/open-lower-drawer" ;
/home/sato/dev/dataset_converter/.venv_n/bin/python main.py --data_name "KS325/open-lower-drawer-r1" ;
/home/sato/dev/dataset_converter/.venv_n/bin/python main.py --data_name "KS325/open-upper-drawer" ;
/home/sato/dev/dataset_converter/.venv_n/bin/python main.py --data_name "KS325/place-doll-lower-r1" ;
/home/sato/dev/dataset_converter/.venv_n/bin/python main.py --data_name "KS325/place-doll-upper-r1"
' > all.log 2>&1 &


