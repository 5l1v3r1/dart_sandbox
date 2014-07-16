import 'dart:html';
import 'dart:math';
import 'dart:async';

void main() {
  new ElastikCarpismaDemosu();
}

Random random = new Random();
num rasgele(num max) => random.nextDouble() * max;

class ElastikCarpismaDemosu {

  Oda oda;
  static const TOP_SAYISI = 20;
  List<Top> toplar = new List<Top>();

  CanvasRenderingContext2D yuzey;

  ElastikCarpismaDemosu() {
    // elastik_carpisma.html içerisindeki "canvas" element'ini getir.
    CanvasElement kanvas = document.querySelector("#canvas");
    yuzey = kanvas.context2D;

    oda = new Oda.fromRect(kanvas.client);

    var renkler = ["green", "orange", "red", "yellow", "blue"];

    // topların ilk üretimi.
    int k = 0;
    while (k < TOP_SAYISI) {

      // rasgele 3 ile 25 arası yarıçap.
      num r = rasgele(22.0) + 3;

      // oda içinde rasgele bir koordinat
      num x = rasgele(oda.en);
      num y = rasgele(oda.boy);

      // -2 +2 arası rasgele hız
      num vx = rasgele(4) - 2.0;
      num vy = rasgele(4) - 2.0;

      // rasgele renk
      String renk = renkler[rasgele(renkler.length).toInt()];

      var top = new Top(r, x, y, vx, vy, renk);
      if (topUygunMu(top)) {
        toplar.add(top);
        k++;
      }
    }
    sahneyiCiz();
  }

  bool topUygunMu(Top yeniTop) {
    if (!oda.iceride(yeniTop)) return false;
    for (Top top in toplar) {
      if (top.topaUzaklik(yeniTop) <= yeniTop.r + top.r) return false;
    }
    return true;
  }

  void sahneyiCiz() {

    // oda alanını temizle
    yuzey.clearRect(oda.x1, oda.y1, oda.x2, oda.y2);
    yuzey.fillStyle = 'rgb(200,200,200)';
    yuzey.fillRect(oda.x1, oda.y1, oda.en, oda.boy);

    // Tüm topları hareket ettir ve çiz.
    for (Top top in toplar) {
      top.duvarCarpmaKontrol(oda);
      top.hareketEt();
      topCiz(top);
    }

    // Topların diğer toplarla çarpışma kontrolü.
    for (int i = 0; i < toplar.length; i++) {
      for (int j = i + 1; j < toplar.length; j++) {
        toplar[i].topCarpmaKontrol(toplar[j]);
      }
    }

    // Sahnenin tekrar çizilmesi isteğini tarayıcıya ilet.
    // Çizim bittiğinde tekrar çizim fonksiyonunu çağır.
    Future<num> f = window.animationFrame;
    f.then((num a) => sahneyiCiz());
  }

  void topCiz(Top top) {
    yuzey.beginPath();
    yuzey.lineWidth = 2;
    yuzey.fillStyle = top.renk;
    yuzey.strokeStyle = top.renk;
    yuzey.arc(top.x, top.y, top.r, 0, PI * 2, false);
    yuzey.fill();
    yuzey.closePath();
    yuzey.stroke();
  }
}

class Oda {

  num x1, y1, x2, y2;

  Oda(this.x1, this.y1, this.x2, this.y2);

  Oda.fromRect(Rectangle rect) {
    x1 = rect.left;
    y1 = rect.top;
    x2 = rect.right;
    y2 = rect.bottom;
  }

  bool iceride(Top ball) => ball.x - ball.r >= x1 && ball.x + ball.r <= x2 && ball.y - ball.r >= y1 && ball.y + ball.r <= y2;

  num get en => (x1 - x2).abs();

  num get boy => (y1 - y2).abs();
}

class Top {

  num r;
  num vX;
  num vY;
  num x;
  num y;
  String renk;

  Top(this.r, this.x, this.y, this.vX, this.vY, this.renk);

  void duvarCarpmaKontrol(Oda oda) {
    if (x - r < oda.x1) {
      vX = -vX;
      x = oda.x1 + r;
    } else if (x + r > oda.x2) {
      vX = -vX;
      x = oda.x2 - r;
    }
    if (y - r < oda.y1) {
      vY = -vY;
      y = oda.y1 + r;
    } else if (y + r > oda.y2) {
      vY = -vY;
      y = oda.y2 - r;
    }
  }

  num topaUzaklik(Top top) => sqrt((top.x - x) * (top.x - x) + (top.y - y) * (top.y - y));

  void topCarpmaKontrol(Top diger) {
    // diger top zaten kendim isem çık.
    if (this == diger) return;
    // çarpışma var mı?
    if (topaUzaklik(diger) <= (r + diger.r)) {
      // formül: v1new = (v1*(m1-m2)+2*m2*v2)/(m1+m2)
      num m1 = r * 0.1;
      num m2 = diger.r * 0.1;
      // kendi vx,vy hızını hemen değiştiremiyoruz. Çünkü değişimden önceki hız değerine
      // diğer denklemde ihtiyacımız var. O yüzden yeni hızı iki geçici değişkene atıyoruz.
      num nX = (vX * (m1 - m2) + (2 * m2 * diger.vX)) / (m1 + m2);
      num nY = (vY * (m1 - m2) + (2 * m2 * diger.vY)) / (m1 + m2);
      diger.vX = (diger.vX * (m2 - m1) + (2 * m1 * vX)) / (m1 + m2);
      diger.vY = (diger.vY * (m2 - m1) + (2 * m1 * vY)) / (m1 + m2);
      vX = nX;
      vY = nY;

      hareketEt();
      diger.hareketEt();
    }
  }

  void hareketEt() {
    x = x + vX;
    y = y + vY;
  }
}
