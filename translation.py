import urllib.request
import json
import re
import argparse


def main(text):
    client_id = "9O__9M4k3AUtPQAMmUsc" 
    client_secret = "Mf_Gyn2W54" 

    def preprocess_sentence(w):
        w = w.strip()

        # 약간 전처리
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r'[ |ㄱ-ㅎ|ㅏ-ㅣ]+', " ", w)

        w = w.strip()

        return w

    text = preprocess_sentence(text)

    encText = urllib.parse.quote(text)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = json.loads(response_body.decode('utf-8'))
        result = result['message']['result']['translatedText']
        print(result)
        return result
    else:
        return "Error Code:" + rescode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text you want to translate into Korean language (wrap your text with "")")
    args = parser.parse_args()

    main(args.text)