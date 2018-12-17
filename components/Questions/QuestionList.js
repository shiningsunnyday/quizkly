import React from 'react';
import { AsyncStorage, Text, FlatList, View, Button, StyleSheet } from 'react-native';
import Question from './Question';

export class QuestionList extends React.Component {

  constructor(props) {
    super(props);
  }

  state = {
    showTrueColor: false,
  }

  changeColor = () => {
    this.setState({ showTrueColor: !this.state.showTrueColor });
  }




  render() {

    // console.log("See if start test works");
    // console.log(this.props.startTest);

    return (
      <View>
        <Button style={styles.button} onPress={this.props.saveList} title="Save this list" />
        <View style={styles.buttons}>

          <Button style={styles.button} onPress={this.props.startTest} title="Test me with this quiz" />
          <Button style={styles.button} onPress={this.changeColor} title="Just show me the answers" />
        </View>
        <FlatList
          showTrueColor={this.state.showTrueColor}
          data={this.props.list}
          renderItem={({item}) =>
            <Question item={item} showTrueColor={this.state.showTrueColor}/>
          }
        />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  buttons: {
    height: 40,
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    backgroundColor: '#aaa'
  },
})
