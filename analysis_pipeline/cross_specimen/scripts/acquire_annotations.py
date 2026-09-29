"""Download checksum-pinned public annotations into the local ignored data directory."""
from pathlib import Path
import hashlib,io,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]

def download(record,out):
    dest=out/record['file']
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==record['sha256']:return dest
    request=urllib.request.Request(record['url'],headers={'User-Agent':'connectome-analysis-pipeline/1.0'})
    with urllib.request.urlopen(request,timeout=90) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError(f"Upstream snapshot changed: {record['file']}; refusing to substitute a different version")
    tmp=dest.with_suffix(dest.suffix+'.download');tmp.write_bytes(data);tmp.replace(dest);return dest

def main():
    out=ROOT/'data_external';out.mkdir(parents=True,exist_ok=True)
    records=json.loads((ROOT/'annotation_sources.json').read_text())
    for record in records:
        path=download(record,out)
        if path.suffix=='.zip':
            name='Sup_Table_7_MatchingCellTypes_final.xlsx'
            with zipfile.ZipFile(path) as z:
                matches=[p for p in z.namelist() if Path(p).name==name]
                if len(matches)!=1:raise ValueError('Expected exactly one published matching workbook')
                target=out/'Nern2025'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(z.read(matches[0]))
        print('Verified',path.name)
if __name__=='__main__':main()
