"""Small dependency-free smoke checks shared by each Python baseline adapter."""
import json
import tempfile
from pathlib import Path
from PIL import Image
from metadata_dataset import load_metadata, prepare_item, restore_a_size, synchronized_preprocess


def run_smoke():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); images=root/'images'; images.mkdir()
        Image.new('RGB',(19,13),'red').save(images/'one_a.png')
        Image.new('RGBA',(19,13),(0,255,0,100)).save(images/'one_b.png')
        Image.new('L',(19,13),127).save(images/'two_a.png')
        Image.new('RGB',(19,13),'blue').save(images/'two_b.png')
        Image.new('RGB',(19,13),'white').save(images/'one_gt.png')
        Image.new('RGB',(19,13),'black').save(images/'two_gt.png')
        data=[
            {'source_index':1,'image':'images/one_gt.png','edit_image':['images/one_a.png','images/one_b.png']},
            {'source_index':2,'image':'images/two_gt.png','edit_image':['images\\two_a.png','images\\two_b.png','images/NEVER_READ_focus_a.png','images/NEVER_READ_focus_b.png']},
            {'source_index':3,'edit_image':['images/missing.png','images/one_b.png']},
            {'source_index':4,'edit_image':['images/one_a.png','images/one_b.png']},
        ]
        metadata=root/'metadata.json'; metadata.write_text(json.dumps(data),encoding='utf-8')
        meta,items=load_metadata(metadata); successes=[]; failures=[]
        for index,item in items:
            try:
                sample=prepare_item(item,index,meta)
                assert sample['a'].mode==sample['b'].mode=='RGB'
                assert restore_a_size(sample['a'],sample).size==(19,13)
                successes.append(index)
            except Exception: failures.append(index)
        assert successes==[0,1,3] and failures==[2], (successes,failures)
        assert prepare_item(data[1],1,meta)['b_path'].name=='two_b.png'
        trained = prepare_item(data[1], 1, meta, mode='train')
        assert trained['target'].getpixel((0, 0)) == (0, 0, 0)
        first = synchronized_preprocess(trained, crop_size=(11, 9), mode='train', seed=7,
                                        hflip=True, vflip=True, rotate90=True)
        second = synchronized_preprocess(trained, crop_size=(11, 9), mode='train', seed=7,
                                         hflip=True, vflip=True, rotate90=True)
        assert first['a'].tobytes() == second['a'].tobytes()
        assert first['a'].size == first['b'].size == first['target'].size
        try:
            prepare_item(data[3], 3, meta, mode='train')
        except ValueError as exc:
            assert "requires 'image'" in str(exc)
        else:
            raise AssertionError('missing GT was accepted in train mode')


if __name__=='__main__': run_smoke(); print('metadata smoke: PASS')
