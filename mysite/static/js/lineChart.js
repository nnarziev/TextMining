var filterYearList = document.getElementById("filter-year");
var filterYearSel = filterYearList.options[filterYearList.selectedIndex].value;
var xAxisData = [];
console.log(filterYearSel);
for (var j = 11; j >= 0; j--) {
    xAxisData.push((filterYearSel - j));
}
var dataset = [];
var colors = ['rgb(255, 99, 132)', 'rgb(255, 159, 64)', 'rgb(138, 174, 20)', 'rgb(53, 197, 226)', 'rgb(196, 49, 226)'];
console.log(data);
var counter = 0;
for (var i = 0; i < 5; i++) {
    var points = [];
    var label = data[counter].text;
    for (var x = 0; x < 12; x++) {
        if (counter < data.length) {
            if (xAxisData[x] === data[counter].year) {
                points.push({x: data[counter].year, y: data[counter].count});
                counter++;
            } else {
                points.push({x: xAxisData[x], y: 0});
            }
        } else {
            points.push({x: xAxisData[x], y: 0});
        }
    }

    dataset.push(
        {
            label: label,
            backgroundColor: colors[i],
            borderColor: colors[i],
            data: points,
            fill: false,
            lineTension: 0
        }
    );

}

new Chart(document.getElementById('trend').getContext('2d'),
    {
        type: 'line',
        data: {
            labels: xAxisData,
            datasets: dataset,
        },
        options: {
            responsive: true,
            tooltips: {
                mode: 'index',
                intersect: false
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                xAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Time'
                    }
                }],
                yAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Frequency'
                    },
                }]
            }
        }
    });