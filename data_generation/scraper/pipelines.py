from genericpath import exists
import os
import re
from itemadapter import ItemAdapter
import zipfile
from settings import FILES_STORE

def get_valid_filename(name):
    s = str(name).strip().replace(' ', '_')
    s = re.sub(r'(?u)[^-\w.]', '', s)
    return s

class FtcLensScraperPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter.get('files'):
            path = adapter['files'][0]['path']
            sku = adapter['sku'][0]
            name = get_valid_filename(adapter['name'][0])

            zip_path = f'{FILES_STORE}/{path}' 
            if exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip:
                    file_in_zip = zip.namelist()[0]
                    
                    unzipped_path = zip.extract(file_in_zip, FILES_STORE)
                    os.rename(unzipped_path, FILES_STORE + '/' + name + '.STEP')

                os.remove(zip_path)
        return item